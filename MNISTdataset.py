from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, Subset
import random
import torch

# Setting seeds for reproducibility
def set_seeds(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For Apple Silicon (MPS)
    if torch.backends.mps.is_available():
        torch.manual_seed(seed)

# Gets dataset of certain size with equal balance amongst the digit labels
def get_balanced_set(dataset, size: int):
    targets = np.array(dataset.targets)
    all_indices = np.arange(len(targets))

    samples_per_class = size//10

    indices = []

    for c in range(10):
        c_indices = all_indices[targets == c]

        indices.extend(np.random.choice(c_indices, size=samples_per_class, replace = False))
    
    return np.array(indices)

# Gives normalized MNIST dataset
def get_dataset():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    return train_dataset

# Gets normalised test dataset
def get_test_loader():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return test_loader

# Returns initial training, validation and pool indices
def get_indices(train_dataset):

    # All the indices of the training data
    total_indices = np.arange(len(train_dataset))

    # Initial Training Set: 20 random but balanced samples
    train_indices = get_balanced_set(train_dataset, 20)

    remaining_indices = np.setdiff1d(total_indices, train_indices)

    # Get the 100 validation datapoints used to optimize the learning rate
    validation_indices = np.random.choice(remaining_indices, size = 100, replace = False)

    # Pool Set: Everything else
    pool_indices = np.setdiff1d(remaining_indices, validation_indices)
    
    return train_indices, validation_indices, pool_indices
