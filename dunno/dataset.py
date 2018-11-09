from torch.utils.data import Dataset
from typing import Tuple
import numpy as np
import torch


def real(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = x**3
    hump = np.exp(-(x - 1)**2/(2*1))  # N(1, 1)
    var = np.ones(x.shape)*3**2 + 16**2*hump
    return y, var


class SomeDataset(Dataset):
    """Dataset of noisy examples from y = x**3."""
    
    def __init__(self):
        x_lim = (-4, 4)
        samples = 1024
        rng = np.random.RandomState(666)
        x_vec = rng.random_sample(samples)*(x_lim[1] - x_lim[0]) + x_lim[0]
        y_vec, y_noise = real(x_vec)
        y_vec += (y_noise**0.5)*rng.standard_normal(samples)
        self.x_vec = x_vec
        self.y_vec = y_vec

    def __len__(self):
        return len(self.x_vec)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x_vec[idx]).unsqueeze(0),
            torch.tensor(self.y_vec[idx]).unsqueeze(0),
        )


def get_dataset():
    return SomeDataset()


def get_data():
    x_lim = (-4, 4)
    samples = 1024
    rng = np.random.RandomState(666)
    x_vec = rng.random_sample(samples)*(x_lim[1] - x_lim[0]) + x_lim[0]
    y_vec, y_noise = real(x_vec)
    y_vec += (y_noise**0.5)*rng.standard_normal(samples)
    return x_vec, y_vec
