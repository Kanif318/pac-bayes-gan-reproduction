import torch
import torch.nn as nn
import torch.nn.functional as F

def _l2_normalize(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return v / (v.norm(p=2) + eps)


@torch.no_grad()
def power_iteration_spectral_norm(
    w: torch.Tensor, u: torch.Tensor, iters: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    u_ = u
    for _ in range(iters):
        v = _l2_normalize(torch.mv(w.T, u_))
        u_ = _l2_normalize(torch.mv(w, v))
    sigma = torch.dot(u_, torch.mv(w, v)).abs()
    return sigma, u_

class BjorckLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.register_buffer("sn_u", _l2_normalize(torch.randn(out_features)))
    
    def forward(self, input):
        sigma, new_u = power_iteration_spectral_norm(self.weight, self.sn_u, iters=1)
        self.sn_u.copy_(new_u)
        w_bar = self.weight / sigma.clamp_min(1.0)
        return F.linear(input, w_bar, self.bias)

class GroupSort(nn.Module):
    def __init__(self, group_size: int = 2):
        super().__init__()
        self.group_size = group_size

    def forward(self, x):
        b, c = x.shape
        if c % self.group_size != 0:
            raise ValueError(
                f"Input channels {c} must be divisible by group_size {self.group_size}"
            )
        x = x.view(b, c // self.group_size, self.group_size)
        x = torch.sort(x, dim=2)[0]
        return x.reshape(b, c)