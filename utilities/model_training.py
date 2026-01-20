'''This module contains functions and classes for model training and evaluation.

- EarlyStopper: Class to monitor validation loss during training and stop the training
    process early if no improvement is observed for a specified number of epochs (patience).
'''
import os
import random
import torch
import numpy as np

from collections import deque


def set_seeds(seed: int):
    '''Set seeds for reproducibility.'''
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

class EarlyStopper:
    '''Class to monitor validation loss during training and stop the training process 
    early if no improvement is observed for a specified number of epochs (patience).

    Args:
        patience (int): Number of consecutive epochs with no improvement after which
            training should be stopped. Default is 1.
        min_delta (float): Minimum change in the validation loss to be considered as 
            an improvement. Default is 0.
    '''
    def __init__(
        self,
        patience: int = 1,
        min_delta: float = 0,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = float('inf')

    def early_stop(self, loss, verbose: bool = True):
        '''Check if training should be stopped early based on the validation loss.
        
        Args:
            loss (float): Current validation loss.
            verbose (bool): Verbosity level. Prints early stopping message if True,
                prints nothing if False. Defaults to True.
        '''
        if loss < self.min_loss:
            self.min_loss = loss
            self.counter = 0
        elif loss >= (self.min_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                if verbose:
                    print(f'Early stopping triggered.')
                return True
        return False


class ReplayBuffer:
    '''Fixed-size buffer to store experience tuples.'''

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition: tuple):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
