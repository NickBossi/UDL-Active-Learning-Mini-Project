import torch
import torch.nn.functional as F

def calc_entropy(preds):
    # Input of shape T x N x dim_output

    # Entropy = Negative sum over C [(1/T sum over T p_c^t) * log(1/T sum over T p_c^t)]
    T,N,C = preds.shape

    # Take mean over T to estimate p(y=c|x,D) leaving an N x C tensor
    mean_probs = torch.mean(preds, dim = 0)     

    log_probs = torch.log(mean_probs+1e-10)

    #sum over the categories, leaving an N dimensional tensor
    entropy = -torch.sum(mean_probs*log_probs, dim =1)     

    return entropy



def calc_BALD(preds):

    entropy = calc_entropy(preds)

    T,N,C = preds.shape
    plogp = preds * torch.log(preds+1e-10)
    sum_TC = torch.sum(plogp, dim  = (0,2)) #sum over T and C, leaving an N dim tensor
    
    MI = entropy + (1/T)*sum_TC

    return MI



def calc_var_rat(preds):
    
    T,N,C = preds.shape


    #mean_preds = torch.mean(preds, dim = 0)

    #max_preds,_ = torch.max(mean_preds, dim =1)


    # Gets all of the indices of the category with the maximum probability for all T x N samples
    max_indices = torch.argmax(preds, dim=2, keepdim = True)
    #print(f"max indices shape: {max_indices.shape}")

    # Fills tensor with zeros
    one_hot = torch.zeros_like(preds, dtype = torch.float)
    #print(f"zeros: {one_hot.shape}")

    # Creates one-hot encoding of probabilities
    one_hot.scatter_(2, max_indices, 1.0)
    #print(f"one_hot scatter: {one_hot.shape}")

    # We now sum over T to get the total number of predictions of each category for a given n
    sum_T = torch.sum(one_hot, dim=0)       # N x C 
    #print(f"sum over T: {sum_T.shape}")

    # Gets the max number for each class for every x 
    f_x,_ = torch.max(sum_T, dim=1)         # N

    return 1.0 - (f_x/T)



def calc_Mean_STD(preds):
    T,N,C = preds.shape

    mean_squared_pred = torch.mean(preds**2, dim =0)        # Mean over T to get MC approximation of expectation, leaving N x C tensor 
    mean_pred_squared = torch.mean(preds, dim=0)**2

    sigmas = torch.sqrt((mean_squared_pred-mean_pred_squared + 1e-6))
    sigmas_c = torch.mean(sigmas, dim=1)

    return sigmas_c



def calc_uniform(preds):
    T,N,C = preds.shape

    return torch.rand(N, device = preds.device) 



def get_TNC_preds(x, model, T, deterministic: bool = False):
    N = x.shape[0]
    
    # If deterministic, simply do a single forward pass without dropout in order to get probabilities
    if deterministic:
        T = 1
        with torch.no_grad():
            output = model(x)

    # Else we use dropout on all T repeats to allow for MC approximation
    else:
        #repeats each x T times and reshapes to shape T*N x other dims
        x_batch = x.unsqueeze(0).expand(T,-1,-1,-1,-1).reshape(T*N, *x.shape[1:])

        with torch.no_grad():
            output = model(x_batch)

    # Reshape to T x N x C ready for the acquisition functions
    logits_TNC = output.reshape(T,N,-1)
    #print(f"Logits shape: {logits_TNC.shape} for determinisic = {str(deterministic)}")

    # Return probabilities associated with logits
    return F.softmax(logits_TNC, dim = -1)
