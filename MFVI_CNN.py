import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from torch.utils.data import DataLoader,Subset
from database import update_database
from feature_CNN import FeatureCNN, train_feature_CNN

class MFVI_CNN(nn.Module):
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
        self.sigma2 = args.sigma2
        self.s2 = args.s2
        self.batch_size = args.batch_size
        self.K = hidden_dim
        self.D = output_dim
        
        # Allows mean and variance to be on same device automatically when calculated for normalisation 
        self.register_buffer("phi_mean", torch.zeros(self.K))
        self.register_buffer("phi_std", torch.ones(self.K))

        # Represent mean and logvariance of posterior, to be optimised with gradient descent
        self.W_mean = nn.Parameter(torch.zeros(self.K, self.D), requires_grad = True)
        self.W_logvar = nn.Parameter(torch.zeros(self.K, self.D), requires_grad = True)

    def forward(self, x):
        phi = self.feature_extractor(x)
        return phi
    
    def loss_fn(self, features, y):
        phi = features.to(self.device)          # [N,K] feature vector
        M = (self.W_mean).to(self.device)       # [K,D] weight means
        N = phi.shape[0]                    

        b = self.b_mean
        b_var = torch.exp(self.b_logvar)

        y_hat = phi @ M +b    # [N,K] x [K,D] = [N,D] 

        S = (torch.exp(self.W_logvar))**2      # [K,D]

        error = torch.sum(((y_hat - y)**2), dim =1)           #Gets L2 norm over D

        epistemic_var = phi**2 @ torch.sum(S, dim = 1)      #[N,K] x [K,1] = [N,1]


        reconstruction_term = -(1/(2*self.sigma2))*torch.sum((error + epistemic_var), dim = 0)       #sums over N

        KL_loss = -0.5 * torch.sum((S + M**2)/self.s2 - 1 + torch.log(self.s2/S))

        loss = -(reconstruction_term+KL_loss)           # Loss is negative ELBO

        return loss
    
    def train_model(self, train_indices= None):

        self.current_epoch = 0
        if train_indices is None:
            train_indices = self.train_indices
        train_loader = DataLoader(Subset(self.train_dataset, train_indices), batch_size = len(train_indices), shuffle=True)

        self.to(self.device)
        self.train()
        self.feature_extractor.eval()

        optimizer = torch.optim.Adam([self.W_mean, self.W_logvar], lr=self.lr)

        e1_loss,e2_loss = 0,0

        # Training loop
        for epoch in range(self.n_epochs):

            epoch_loss = 0.0
            n = 0

            for data, target in train_loader:
                data = data.to(self.device).float()

                # Change predictions to one-hot encodings
                target = (F.one_hot(target, num_classes = 10)).to(self.device).float()

                optimizer.zero_grad()

                # Gets features and normalises
                features = (self.feature_extractor(data) - self.phi_mean)/self.phi_std

                loss = self.loss_fn(features, target)
                #print(loss)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * target.size(0)
                n+=target.size(0)
            mean_loss = epoch_loss / n

            # Stops training if loss converges
            e2_loss = mean_loss
            delta = np.abs(e1_loss - e2_loss)
            if delta <1e-5:
                print(f"Converged, stopping training at epoch {epoch}")
                break
            else:
                e1_loss = mean_loss

    def calc_predictive(self,x_star):
        phi = (self.feature_extractor(x_star) - self.phi_mean) / self.phi_std
        
        M = (self.W_mean).to(self.device)
        S = torch.exp(self.W_logvar.to(self.device))           # [K, D]

        pred_mean = phi @ M     # [N,K] @ [k,D] = [N,D]

        pred_var = phi**2 @ S        #[N,K] @ [K,D] = [N,D]
        
        pred_var = pred_var.sum(dim=1)              # Sum over dimensions to get a single variance per data point

        return pred_mean, pred_var
    
    def calc_pred_var(self, pool_indices):
        X_star_loader = DataLoader(Subset(self.train_dataset, pool_indices), batch_size = self.batch_size, shuffle = False)
        
        variances = []

        self.eval()
        with torch.no_grad():
            for batch_idx, (x_star,_) in enumerate(X_star_loader):
                x_star = x_star.to(self.device)
                _, pred_var = self.calc_predictive(x_star)      

                variances.append(pred_var.cpu())

        return torch.cat(variances).numpy()
    
    def test_model(self):
        
        total_loss = 0.0
        n = 0
        self.eval()
        self.to(self.device).eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data = data.to(self.device)

                target = (F.one_hot(target, num_classes = self.D)).float().to(self.device)

                pred_mean, _ = self.calc_predictive(data)

                loss = nn.MSELoss(reduction = 'sum')(pred_mean, target)

                total_loss += loss.item()
                n+= target.numel()

            return np.sqrt(total_loss/n)        #returns RMSE

    def normalise_features(self, train_indices):
        train_loader = DataLoader(Subset(self.train_dataset, train_indices), batch_size = self.batch_size, shuffle = False)

        features = []
        self.feature_extractor.eval()

        for x,_ in train_loader:
            x = x.to(self.device)
            phi = self.feature_extractor(x)
            features.append(phi.detach())

        features = torch.cat(features, dim = 0)
        self.phi_mean.copy_(features.mean(dim = 0))
        self.phi_std.copy_(features.std(dim = 0, unbiased = False).clamp_min(1e-6))

def MFVI_acquisition(args,run_num, feature_CNN, retrain: bool = True):
    train_indices = args.train_indices.copy()
    pool_indices = args.pool_indices.copy()

    # Initialises MFVI_CNN and copies over train_feature_CNN
    model = MFVI_CNN(args).to(args.device)
    model.feature_extractor = feature_CNN.feature_extractor

    # Freezes feature parameters
    for p in model.feature_extractor.parameters():
        p.requires_grad = False
    
    model.normalise_features(train_indices)

    model.train_model()

    print(f"RMSE of base model = {model.test_model()}")
    run = []
    for i in range(args.n_acq):
        acq_step = (i+1)*10

        uncertainty_scores = model.calc_pred_var(pool_indices)
        print(f"Uncertainty_scores = {uncertainty_scores}")
        acquired_pool_indices = np.argsort(uncertainty_scores)[::-1][:10]


        acquired_dataset_indices = pool_indices[acquired_pool_indices]

        pool_indices = np.setdiff1d(pool_indices, acquired_dataset_indices)
        train_indices = np.concatenate((train_indices, acquired_dataset_indices))

        if retrain:
            print(f"Retraining feature extractor at acq step {acq_step}")

            feature_CNN = FeatureCNN().to(args.device)
            feature_CNN = train_feature_CNN(model = feature_CNN, train_dataset=args.train_dataset, train_indices = train_indices, device =  args.device, n_epochs = 50)
            feature_CNN.eval()

            # Reinitialise and train model
            model = MFVI_CNN(args).to(args.device)
            model.feature_extractor = feature_CNN.feature_extractor

            model.feature_extractor.eval()
            for p in feature_CNN.feature_extractor.parameters():
                p.requires_grad = False
        
            model.normalise_features(train_indices)

        model.train_model(train_indices)
        RMSE = model.test_model()

        # Adds run to dataset
        data = [retrain, "MFVI_inf", run_num, acq_step, RMSE]

        update_database(data = data, filename = "inference_database")

        print(f"RMSE for run {run_num} of MFVI_CNN at acq-step {acq_step} = {RMSE}")
    
    return model

def run_MFVI_experiment(args, run_nums , feature_CNN, retrain: bool = True):

    for run_num in run_nums:
        model = MFVI_acquisition(args, run_num, feature_CNN, retrain)
        print(f"Test accuracy for MFVI_CNN = {model.test_model()}")

        


