import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Gaussian(nn.Module):
    def __init__(self, mu, rho, fixed=False):
        super().__init__()
        self.mu = nn.Parameter(mu, requires_grad=not fixed)
        self.rho = nn.Parameter(rho, requires_grad=not fixed)

    @property
    def sigma(self):
        return F.softplus(self.rho)

    def sample(self):
        epsilon = torch.randn_like(self.sigma)
        return self.mu + self.sigma * epsilon

    def compute_kl(self, other):
        b1 = torch.pow(self.sigma, 2)
        b0 = torch.pow(other.sigma, 2)
        term1 = torch.log(torch.div(b0, b1))
        term2 = torch.div(torch.pow(self.mu - other.mu, 2), b0)
        term3 = torch.div(b1, b0)
        kl_div = (torch.mul(term1 + term2 + term3 - 1, 0.5)).sum()
        return kl_div

class ProbLinLayer(nn.Module):
    def __init__(self, in_features, out_features, rho_prior, rho_init, bias=True):
        super(ProbLinLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kl_div = 0

        k = 1 / np.sqrt(self.in_features)
        
        weight_mu = torch.FloatTensor(out_features, in_features).uniform_(-k, k)
        bias_mu = torch.FloatTensor(out_features).uniform_(-k, k)
        weight_rho = torch.ones(out_features, in_features) * rho_init
        bias_rho = torch.ones(out_features) * rho_init

        self.weight = Gaussian(weight_mu, weight_rho, fixed=False)
        self.bias = Gaussian(bias_mu, bias_rho, fixed=False)

        prior_weight_mu = weight_mu.clone()
        prior_bias_mu = bias_mu.clone()
        prior_weight_rho = torch.ones(out_features, in_features) * rho_prior
        prior_bias_rho = torch.ones(out_features) * rho_prior

        self.prior_weight = Gaussian(prior_weight_mu, prior_weight_rho, fixed=True)
        self.prior_bias = Gaussian(prior_bias_mu, prior_bias_rho, fixed=True)

    def forward(self, input, sample=False):
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        
        if self.training:
            self.kl_div = self.kl_divergence()
            
        return F.linear(input, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        weight_kl = self.weight.compute_kl(self.prior_weight)
        bias_kl = self.bias.compute_kl(self.prior_bias)
        return weight_kl + bias_kl