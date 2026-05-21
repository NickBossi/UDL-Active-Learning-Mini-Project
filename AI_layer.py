import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from torch.utils.data import DataLoader,Subset
from database import update_database
from feature_CNN import FeatureCNN, train_feature_CNN
from regression_CNN import reg_CNN


# Class for active inference
class AILayer:
    def __init__(self, args, feature_CNN, feature_dim = 128):
        self.sigma2 = args.sigma2
        self.s2 = args.s2
        self.K = feature_dim
        self.D = args.num_classes
        self.device = args.device
        self.CNN = feature_CNN.to(self.device)
        self.batch_size = args.batch_size
        self.train_dataset = args.train_dataset
        self.test_loader = args.test_loader
        
    # Fits posterior using current training set
    def fit_posterior(self, train_indices):

        for batch_idx, (X,targets) in enumerate(DataLoader(Subset(self.train_dataset, train_indices), batch_size = len(train_indices))):
            X = X.to(self.device)
            targets = F.one_hot(targets, num_classes = self.D).float().to(self.device)
            
            # Gets features using feature extractor
            phi, _  = self.CNN(X)

            self.Sigma_inv = (1/self.sigma2) * (phi.T @ phi) + (torch.eye(self.K, device = self.device) / self.s2)

            self.Sigma_post = torch.inverse(self.Sigma_inv).to(self.device)

            self.W_mean = ((1/self.sigma2) * self.Sigma_post @ (phi.T @ targets)).to(self.device)

    # Calulcates predictive mean and variance
    def calc_predictive(self, x_star):
        self.CNN.eval()

        phi,_ = self.CNN(x_star.to(self.device))

        M = (self.W_mean).to(self.device)

        pred_mean = phi @ M

        epistemic_variance = torch.einsum('ni,ij,nj->n', phi, self.Sigma_post, phi)

        pred_var = self.sigma2 + epistemic_variance

        return pred_mean, pred_var
    
    # Returns all the predictive variances for the acquisition pool
    def calc_pred_var(self, pool_indices):
        X_star_loader = DataLoader(Subset(self.train_dataset, pool_indices), batch_size = self.batch_size, shuffle = False)
        
        variances = []

        with torch.no_grad():
            for batch_idx, (x_star,_) in enumerate(X_star_loader):
                x_star = x_star.to(self.device)
                _, pred_var = self.calc_predictive(x_star)      

                variances.append(pred_var.cpu())

        return torch.cat(variances).numpy()
    
    def test_model(self):
        
        total_loss = 0.0
        n = 0

        self.CNN.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data = data.to(self.device)

                target = (F.one_hot(target, num_classes = self.D)).float().to(self.device)

                pred_mean, _ = self.calc_predictive(data)

                loss = nn.MSELoss(reduction = 'sum')(pred_mean, target)

                total_loss += loss.item()
                n+= target.numel()

            return np.sqrt(total_loss/n)        #returns RMSE
    

def AI_acquisition(args, run_num, feature_CNN, retrain: bool = False, train_reg: bool = False):
    train_indices = args.train_indices.copy()
    pool_indices = args.pool_indices.copy()

    # Freezes parameters
    for p in feature_CNN.feature_extractor.parameters():
        p.requires_grad = False

    # Initialises the Analytic Inference Layer
    AI_layer = AILayer(args, feature_CNN)
    AI_layer.fit_posterior(train_indices)

    for i in range(args.n_acq):
        acq_step = (i+1)*10

        uncertainty_scores = AI_layer.calc_pred_var(pool_indices)

        acquired_pool_indices = np.argsort(uncertainty_scores)[::-1][:10]

        acquired_dataset_indices = pool_indices[acquired_pool_indices]

        pool_indices = np.setdiff1d(pool_indices, acquired_dataset_indices)
        train_indices = np.concatenate((train_indices, acquired_dataset_indices))

        # If retrain is true, train feature CNN on new data to improve feature extraction 
        if retrain:
            feature_CNN = FeatureCNN().to(args.device)
            feature_CNN = train_feature_CNN(feature_CNN, args.train_dataset, args.device, train_indices, n_epochs = 50)
            for p in feature_CNN.feature_extractor.parameters():
                p.requires_grad = False
            AI_layer = AILayer(args, feature_CNN)
        
        #Fits posterior to new dataset
        AI_layer.fit_posterior(train_indices)

        # Gets test scores under new posterior
        RMSE = AI_layer.test_model()

        # Updates database with new test score
        data = [retrain, "analytic_inf", run_num, acq_step, RMSE]
        update_database(data = data, filename = "inference_database")

        print(f"RMSE for run {run_num} of inference_CNN at acq-step {acq_step} = {RMSE}")
    
    return AI_layer

def run_AI_experiment(args, run_nums, feature_CNN, retrain: bool = False, train_reg: bool = False):
    for run_num in run_nums:
        model = AI_acquisition(args, run_num, feature_CNN, retrain, train_reg)
        print(f"Test accuracy for analytic_inf = {model.test_model()}")












