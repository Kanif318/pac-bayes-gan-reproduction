import torch.nn as nn
from .layers import BjorckLinear, GroupSort
from .prob_nn import ProbLinLayer

class DeterministicGenerator(nn.Module):
    def __init__(self, z_dim, hidden_dim, out_dim=2):
        super().__init__()
        self.latent_dim = z_dim
        self.layer_1 = nn.Linear(z_dim, hidden_dim)
        self.activation_1 = nn.ReLU(True)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation_2 = nn.ReLU(True)
        self.layer_3 = nn.Linear(hidden_dim, hidden_dim)
        self.activation_3 = nn.ReLU(True)
        self.layer_4 = nn.Linear(hidden_dim, out_dim)

    def forward(self, z):
        out = self.activation_1(self.layer_1(z))
        out = self.activation_2(self.layer_2(out))
        out = self.activation_3(self.layer_3(out))
        out = self.layer_4(out)
        return out

class ProbGenerator(nn.Module):
    def __init__(self, z_dim, hidden_dim, out_dim=2, rho_init=-5, rho_prior=-5):
        super().__init__()
        self.latent_dim = z_dim
        self.layer_1 = ProbLinLayer(z_dim, hidden_dim, rho_prior, rho_init)
        self.activation_1 = nn.ReLU(True)
        self.layer_2 = ProbLinLayer(hidden_dim, hidden_dim, rho_prior, rho_init)
        self.activation_2 = nn.ReLU(True)
        self.layer_3 = ProbLinLayer(hidden_dim, hidden_dim, rho_prior, rho_init)
        self.activation_3 = nn.ReLU(True)
        self.layer_4 = ProbLinLayer(hidden_dim, out_dim, rho_prior, rho_init)
        self.kl_div = 0

    def forward(self, z, sample=True):
        out = self.activation_1(self.layer_1(z, sample))
        out = self.activation_2(self.layer_2(out, sample))
        out = self.activation_3(self.layer_3(out, sample))
        out = self.layer_4(out, sample)
        
        if self.training:
            self.kl_div = self.kl_divergence()
        return out

    def kl_divergence(self):
        return (
            self.layer_1.kl_divergence()
            + self.layer_2.kl_divergence()
            + self.layer_3.kl_divergence()
            + self.layer_4.kl_divergence()
        )

    def init_from_deterministic(self, det: "DeterministicGenerator"):
        pairs = [
            (self.layer_1, det.layer_1),
            (self.layer_2, det.layer_2),
            (self.layer_3, det.layer_3),
            (self.layer_4, det.layer_4),
        ]
        for prob_layer, det_layer in pairs:
            prob_layer.weight.mu.data.copy_(det_layer.weight.data)
            prob_layer.bias.mu.data.copy_(det_layer.bias.data)
            prob_layer.prior_weight.mu.data.copy_(det_layer.weight.data)
            prob_layer.prior_bias.mu.data.copy_(det_layer.bias.data)

class Critic(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.layer_1 = BjorckLinear(in_dim, hidden_dim)
        self.activation_1 = GroupSort(2)
        self.layer_2 = BjorckLinear(hidden_dim, hidden_dim)
        self.activation_2 = GroupSort(2)
        self.layer_3 = BjorckLinear(hidden_dim, hidden_dim)
        self.activation_3 = GroupSort(2)
        self.layer_4 = BjorckLinear(hidden_dim, 1)

    def forward(self, x):
        out = self.activation_1(self.layer_1(x))
        out = self.activation_2(self.layer_2(out))
        out = self.activation_3(self.layer_3(out))
        out = self.layer_4(out)
        return out