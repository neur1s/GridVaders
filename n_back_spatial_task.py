from scipy import stats
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch


# Dict to convert movement idx to actual movement coordinates
idx2mov = {0:np.array([1,0], dtype=int), 
           1:np.array([-1,0], dtype=int), 
           2:np.array([0,1], dtype=int), 
           3:np.array([0,-1], dtype=int)}

# Convert coordinate to flattened idx
def loc2idx(loc, grid_size=np.array([5, 5], dtype=int)):
    return loc[0]*grid_size[0] + loc[1]

# Convert location flattened idx to coordinate
def idx2loc(idx, grid_size=np.array([5, 5], dtype=int)):
    return np.array([idx // grid_size[0], idx % grid_size[0]], dtype=int)


def sample_n_back_spatial(n, max_length=40, grid_size=np.array([5, 5], dtype=int), boundary='periodic', return_trajectory=False):
    """
    Function to generate a sample for the n-back spatial task.

    Args:
    - n: response delay
    - p_stop: after n steps, probability of stoping walk (default=0.05)
    - max_length: maximum trajectory length (left zero-padding is applied to reach this length)
    - grid_size (array-like): size of gridworld, must be odd (default=[5,5])
    - boundary ['periodic', 'strict']: boundary conditions
    - return_trajectory (bool): whether to return trajectory

    Returns: movements (1D array, as index), n_back_idx (n-back location as idx), (trajectory) 
    
    """
    assert boundary in ['periodic', 'strict'], "boundary must be either 'periodic' or 'strict'"
    assert (grid_size[0] % 2 == 1) & (grid_size[1] % 2 == 1), "grid size must be odd"

    zero = np.array([(grid_size[0]-1)//2, (grid_size[1]-1)//2], dtype=int)
    movements = np.random.randint(4, size=max_length)
    
    trajectory = [zero]
    
    for idx in movements:
        if boundary == 'periodic':
            trajectory.append((trajectory[-1] + idx2mov[idx]) % grid_size)
        elif boundary == 'strict':
            trajectory.append(np.clip(trajectory[-1] + idx2mov[idx], a_min=[0,0], a_max=grid_size-1))
        
    trajectory = np.array(trajectory)

    n_back_idx = np.array([loc2idx(trajectory[i], grid_size=grid_size) for i in range(max_length-n)])

    if return_trajectory:
        return movements, n_back_idx, trajectory
    else:
        return movements, n_back_idx
        

# Dataset class for storing inputs/outputs
class NBackDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)
		
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# Wraper to create datasets
def create_n_back_dataset(num_samples, n, max_length=40, grid_size=np.array([5, 5], dtype=int), boundary='periodic'):
    X, Y = [], []
    for _ in range(num_samples):
        x, y = sample_n_back_spatial(n, max_length=max_length, grid_size=grid_size, boundary=boundary)
        X.append(x); Y.append(y)

    X = np.vstack(X)
    Y = np.vstack(Y)

    X = torch.tensor(X, dtype=int)
    Y = torch.tensor(Y, dtype=int)

    return NBackDataset(X, Y)
