import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np 
import tqdm
import copy
from MNISTdataset import set_seeds
from acquisition_functions import calc_entropy, calc_BALD, calc_var_rat, calc_Mean_STD, calc_uniform, get_TNC_preds, mean_change
from database import update_database

class CNN(nn.Module):
    def __init__(self, 
                args,
                num_filters: int = 32, 
                hidden_dim: int = 128,
                kernel_size: int = 4,
                max_pool: int = 2,
                output_dim:int=10,
                width: int = 28,
                height: int = 28
                ):
        super().__init__()
        self.train_dataset = args.train_dataset
        self.train_indices, self.validation_indices, self.pool_indices = args.train_indices, args.validation_indices, args.pool_indices
        self.test_loader = args.test_loader
        self.device = args.device
        self.wd = args.wd
        self.lr = args.lr
        self.n_epochs = args.n_epochs
        self.batch_size = args.batch_size

        self.layers = nn.Sequential(
            # Convolution 1
            nn.Conv2d(in_channels = 1, out_channels = num_filters, kernel_size = kernel_size),
            nn.ReLU(),

            # Convolution 2
            nn.Conv2d(in_channels = num_filters, out_channels = num_filters, kernel_size = kernel_size),
            nn.ReLU(),

            # Max pooling
            nn.MaxPool2d(kernel_size=max_pool),

            # Dropoout 1
            nn.Dropout(p=0.25),

            # Flatten and fully connected (account for two convolutions and maxpooling in input dimension)
            nn.Flatten(),
            nn.Linear(in_features = (num_filters 
                                    * ((width - 2 * kernel_size + 2) // max_pool)
                                    * ((height - 2 * kernel_size + 2) // max_pool)), 
                      out_features = hidden_dim),
            nn.ReLU(),

            # Droput 2
            nn.Dropout(p=0.5),

        )
        self.final_layer = nn.Linear(in_features = hidden_dim, out_features = output_dim)

    def forward(self, x):
        self.h = self.layers(x)
        x = self.final_layer(self.h)
        return x

    def train_model(self, train_indices = None):
        if train_indices is None:
            train_indices = self.train_indices

        train_loader = DataLoader(Subset(self.train_dataset, train_indices), batch_size=128, shuffle=True)

        self.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay = self.wd)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(self.n_epochs):
            for data, target in train_loader:
                data = data.to(self.device)
                target = target.to(self.device)

                optimizer.zero_grad()
                output = self(data)
                
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        return self

    def validation(self):
        print(f"Performing validation.")
        
        val_loader = DataLoader(Subset(self.train_dataset, self.validation_indices), batch_size = len(self.validation_indices), shuffle = False)

        val_losses = []
        val_accuracies = []
        lambdas = [np.exp(-i) for i in np.linspace(0,5,40)]


        for wd in lambdas:
            set_seeds(2025)
            
            model_copy = copy.deep_copy(self)
            model_copy.train_model(weight_decay = wd)
            model_copy.eval()

            correct = 0

            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(self.device).float(), target.to(self.device)
                output = model_copy(data)

                _, predicted = torch.max(output, dim=1)
                correct += (target == predicted).sum().item()

                val_accuracies.append(correct/len(len(self.validation_indices)))

                val_loss = nn.CrossEntropyLoss()(output, target)
                val_losses.append(val_loss.item())

        plt.plot(lambdas, val_losses)
        plt.show()

        opt_wd = lambdas[np.argmax(val_accuracies)]
        print(f"Validation complete!")

        return opt_wd 

    def test_model(self, deterministic: bool = False, T: int = 20):

        correct = 0
        total = 0

        # If deterministic, no dropout at test time: 
        if deterministic:
            self.eval()
            with torch.no_grad():
                for batch_idx, (data, target) in (pbar:= tqdm.tqdm(enumerate(self.test_loader))):
                    data, target = data.to(self.device).float(), target.to(self.device)
                    output = self(data)
                    _, predicted = torch.max(output, dim=1)
                    correct += (target == predicted).sum().item()
                    total +=target.size(0)

        # Else: do MC over dropout
        else:
            self.train()
            with torch.no_grad():
                for batch_idx, (data, target) in (pbar:= tqdm.tqdm(enumerate(self.test_loader))):
                    data, target = data.to(self.device).float(), target.to(self.device)

                    # Get T predictions for every input
                    TNC_preds = get_TNC_preds(data, self,T)

                    # Take mean over T to get MC estimate of model output
                    MC_estimate = torch.mean(TNC_preds, dim =0)

                    # Get predicted class and check if correct
                    _, predicted = torch.max(MC_estimate, dim=1)
                    correct += (target == predicted).sum().item()
                    total +=target.size(0)

        accuracy = correct/total
        return accuracy

# Acquisition loop
def train_w_acquisition(args,
                        base_model, 
                        acq_fn_name, 
                        acq_fn, 
                        opt_wd = None,
                        n_acq: int = 100, 
                        acq_batch_size = 32,
                        run_num: int = 0,
                        deterministic: bool = False,
                        k: int = 10):
    # Ensures base_model and indices are the same at the start of each acquisition function

    model = copy.deepcopy(base_model)              
    train_indices = args.train_indices.copy()
    pool_indices = args.pool_indices.copy()
    T = args.T

    if opt_wd is not None:
        args.wd = opt_wd

    # Sets rng for shuffling pool dataset, necessary for when there are tied acquisition scores.

    rng = np.random.default_rng()
    # Iterates through the acquisition steps, evaluating the entire pool against the relevant
    # acquisition function, and choosing the top 10 samples, adds these to the training set and further trains
    # the model 

    for i in range(n_acq):

        acq_step = (i+1)*10
        
        uncertainty_scores = []

        # Gets data from pool from which we will acquire new xs
        pool_data = DataLoader(Subset(args.train_dataset, pool_indices), batch_size = acq_batch_size, shuffle = False)

        print(f"Calulating uncertainty scores using: {acq_fn_name}, acq_step: {acq_step}")
        if deterministic:
            model.eval()
        else:
            model.train()

        # Gets uncertainty scores in batches due to memory constraints
        for batch_idx, (x_batch,_) in enumerate(pool_data):
            x_batch = x_batch.to(args.device)

            # EMC requires different arguments and does not use MC-dropout
            if acq_fn_name == "mean_change":
                acq_values = acq_fn(x_batch, model)
            else:
                acq_values = acq_fn(get_TNC_preds(x_batch, model, T, deterministic))

            # Helps with memory
            if isinstance(acq_values, torch.Tensor):
                scores = acq_values.detach().cpu().numpy()
            else:
                scores = acq_values
                
            uncertainty_scores.append(scores)
            
            # Delete unused values to help clear memory
            del x_batch, acq_values 

        # Combines all uncertainty scores into one array
        uncertainty_scores = np.concatenate(uncertainty_scores)

        # First randomly shuffles indices so that ties aren't chosen deterministically
        perm = rng.permutation(len(uncertainty_scores))

        # Then takes the top k of these and maps it back to the original indices
        acquired_pool_indices = perm[np.argsort(uncertainty_scores[perm], kind = "stable")[::-1][:k]]

        # Maps these indices back to the actual training set indices and updates pool and training indices
        acquired_dataset_indices = pool_indices[acquired_pool_indices]
        pool_indices = np.setdiff1d(pool_indices, acquired_dataset_indices)
        train_indices = np.concatenate((train_indices, acquired_dataset_indices))
 
        # Sets different seed for each run to get mean and std statistics
        set_seeds(2025+run_num)

        model = CNN(args = args).to(args.device)
        model.train_model(train_indices = train_indices)
        accuracy = model.test_model(deterministic, T = args.T)
        
        # Svae data for plotting
        data = [deterministic, run_num, acq_fn_name, acq_step, accuracy]
        update_database(data)

        print(f"Accuracy for acq_fn {acq_fn_name} at acq-step {acq_step} is {accuracy}")
    
    return model

# Helps run the three runs for statistics
def run_experiments(args,
                    acq_fns: dict, 
                    deterministic: bool = False,
                    run_nums: list = [0,1,2]):
    
    set_seeds(2025)

    # Initialises new model
    model = CNN(args).to(args.device)

    # Gets optimal wd via validation, setting seeds to ensure all acq_fns and runs have same base model
    opt_wd = model.validation()
    args.wd = opt_wd
    print(f"Opt wd: {opt_wd}")

    # Trains base model with optimal wd
    base_model = CNN(args).to(args.device).train_model()
    print(f"Base model test accuracy: {base_model.test_model(deterministic = deterministic, T = args.T)}")

    # Iterates through acquisition functions, performing active learning and saving acquired data in database
    for key, value in acq_fns.items():
        for run_num in run_nums:
            model = train_w_acquisition(args = args,
                                        base_model=base_model, 
                                        acq_fn_name = key, 
                                        acq_fn = value, 
                                        opt_wd = opt_wd,
                                        n_acq=100, 
                                        run_num = run_num,
                                        deterministic = deterministic)
            accuracy = model.test_model(deterministic = deterministic, T = args.T)

        print(f"Test accuracy for acquisition function: {key} = {accuracy}")