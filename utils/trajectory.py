import torch
import torch.nn.functional as F


__all__ = ['BalancedRDTrajectoryState']

class BalancedRDTrajectoryState:
    """
    Keeps the state for Solution 1: only handles xi and weights update.
    """
    def __init__(self, device, beta=0.025, gamma=0.001):
        self.xi = torch.zeros(2, device=device)
        self.weights = F.softmax(self.xi, dim=0)
        self.beta = beta
        self.gamma = gamma

    def update(self, loss_rate, loss_dist, loss_rate_next, loss_dist_next):
        """
        Update weights given current/next rate and distortion losses (scalars).
        """
        delta = torch.stack([
            torch.log(loss_rate.detach() + 1) - torch.log(loss_rate_next.detach() + 1),
            torch.log(loss_dist.detach() + 1) - torch.log(loss_dist_next.detach() + 1)
        ]).to(self.xi.device)
        self.xi = self.xi - self.beta * (delta + self.gamma * self.xi)
        self.weights = F.softmax(self.xi, dim=0)