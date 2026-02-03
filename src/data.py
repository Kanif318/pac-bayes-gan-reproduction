import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

class RingGMMTrunc:
    def __init__(self, n_comps=8, radius=3, max_radius=3.2, sigma=0.05, n_samples=10000, batch_size=256):
        self.n_comps = n_comps
        self.radius = radius
        self.max_radius = max_radius
        self.sigma = sigma
        self.batch_size = batch_size

        centers_x = [self.radius * np.cos((2 * np.pi * i) / self.n_comps) for i in range(self.n_comps)]
        centers_y = [self.radius * np.sin((2 * np.pi * i) / self.n_comps) for i in range(self.n_comps)]
        
        self.means = torch.tensor([[cx, cy] for cx, cy in zip(centers_x, centers_y)], dtype=torch.float32)
        
        indices = torch.randint(0, self.n_comps, (n_samples * 2,)) 
        centers = self.means[indices]
        noise = torch.randn(n_samples * 2, 2) * self.sigma
        data = centers + noise
        
        mask = (data[:, 0]**2 + data[:, 1]**2 <= self.max_radius**2)
        data = data[mask][:n_samples]
        
        n_train = int(0.8 * len(data))
        self.train_data = data[:n_train]
        self.test_data = data[n_train:]
        
        self.train_loader = DataLoader(TensorDataset(self.train_data, torch.zeros(n_train)), batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(TensorDataset(self.test_data, torch.zeros(len(self.test_data))), batch_size=batch_size, shuffle=False)

    def get_loaders(self):
        return self.train_loader, self.test_loader

class GridGMMTrunc:
    def __init__(self, comps=5, side=8, max_side=8.2, sigma=0.05, n_samples=10000, batch_size=256):
        self.comps = comps
        self.side = side
        self.max_side = max_side
        self.sigma = sigma
        
        centers_x = [-self.side / 2 + i * self.side / (self.comps - 1) for i in range(self.comps)]
        centers_y = [-self.side / 2 + j * self.side / (self.comps - 1) for j in range(self.comps)]
        
        means = [[cx, cy] for cx in centers_x for cy in centers_y]
        self.means = torch.tensor(means, dtype=torch.float32)
        
        n_total_comps = self.comps * self.comps
        indices = torch.randint(0, n_total_comps, (n_samples * 2,))
        centers = self.means[indices]
        noise = torch.randn(n_samples * 2, 2) * self.sigma
        data = centers + noise
        
        margin = max_side / 2
        mask_x = (data[:, 0] >= -margin) & (data[:, 0] <= margin)
        mask_y = (data[:, 1] >= -margin) & (data[:, 1] <= margin)
        data = data[mask_x & mask_y][:n_samples]
        
        n_train = int(0.8 * len(data))
        self.train_data = data[:n_train]
        self.test_data = data[n_train:]
        
        self.train_loader = DataLoader(TensorDataset(self.train_data, torch.zeros(n_train)), batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(TensorDataset(self.test_data, torch.zeros(len(self.test_data))), batch_size=batch_size, shuffle=False)

    def get_loaders(self):
        return self.train_loader, self.test_loader