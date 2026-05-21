import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from torch.utils.data import DataLoader,Subset
from database import update_database


class reg_CNN(nn.Module):
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
        self.D = args.num_classes

        self.Layers = nn.Sequential(
            # Convolution 1
            nn.Conv2d(in_channels = 1, out_channels = num_filters, kernel_size = kernel_size),
            nn.ReLU(),

            # Convolution 2
            nn.Conv2d(in_channels = num_filters, out_channels = num_filters, kernel_size = kernel_size),
            nn.ReLU(),

            # Max pooling
            nn.MaxPool2d(kernel_size=max_pool),


            # Flatten and fully connected (account for two convolutions and maxpooling in input dimension)
            nn.Flatten(),
            nn.Linear(in_features = (num_filters 
                                    * ((width - 2 * kernel_size + 2) // max_pool)
                                    * ((height - 2 * kernel_size + 2) // max_pool)), 
                      out_features = hidden_dim),
            nn.ReLU(),

            # Fully connected 2 
            nn.Linear(in_features = hidden_dim, out_features = output_dim)
        )


    def forward(self, x):
        return self.Layers(x)
    

    def train_model(self, train_indices= None):
        self.current_epoch = 0
        if train_indices is None:
            train_indices = self.train_indices
        train_loader = DataLoader(Subset(self.train_dataset, train_indices), batch_size=self.batch_size, shuffle=True)

        # Initialize model, optimizer and loss function 
        self.to(self.device)
        self.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay = self.wd)

        # Training loop
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0


            for data, target in train_loader:
                data = data.to(self.device).float()

                # Change predictions to one-hot encodings
                target = (F.one_hot(target, num_classes = 10)).to(self.device).float()

                optimizer.zero_grad()
                # Extract prediction
                model_outputs = self(data)
                
                loss = nn.MSELoss()(model_outputs, target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            #print(f"epoch loss = {epoch_loss}")
            self.current_epoch +=1

    def test_model(self):
        
        total_loss = 0.0
        n = 0

        self.to(self.device).eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data = data.to(self.device)

                target = (F.one_hot(target, num_classes = self.D)).float().to(self.device)

                output = self(data)

                loss = nn.MSELoss(reduction = 'sum')(output, target)

                total_loss += loss.item()

                #print(f"Target shape = {target.shape}")
                #print(f"Target.numel = {target.numel()}")
                n+= target.numel()

            return np.sqrt(total_loss/n)        #returns RMSE

def reg_acquisition(args, run_num = 0):
    train_indices = args.train_indices.copy()
    pool_indices = args.pool_indices.copy()

    #Trains regression on initial dataset
    model = reg_CNN(args)
    model.train_model()
    print(f"Base model test = {model.test_model()}")

    for i in range(args.n_acq):
        acq_step = (i+1)*10


        acquired_dataset_indices = np.random.choice(pool_indices, size = 10, replace = False)
        #print(f"Acquired dataset indices = {acquired_dataset_indices}")

        pool_indices = np.setdiff1d(pool_indices, acquired_dataset_indices)
        train_indices = np.concatenate((train_indices, acquired_dataset_indices))

        # Reinitializes model and trains on new dataset
        model = reg_CNN(args)
        model.train_model(train_indices)
        RMSE = model.test_model()

        data = [False, "uniform", run_num, acq_step, RMSE]
        update_database(data = data, filename = "inference_database")

        print(f"RMSE for run {run_num} of uniform sampling at acq-step {acq_step} = {RMSE}")
    
    return model

def run_reg_experiment(args, run_nums):
    opt_wd= 1e-4

    for run_num in run_nums:
        model = reg_acquisition(args, run_num)
        print(f"Test accuracy for random regression = {model.test_model()}")













