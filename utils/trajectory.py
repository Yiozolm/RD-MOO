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
        # Step 1: log-diff vector
        v = torch.stack([
            torch.log(loss_rate.detach()) - torch.log(loss_rate_next.detach()),
            torch.log(loss_dist.detach()) - torch.log(loss_dist_next.detach())
        ]).to(self.xi.device)
        # Step 2: Jacobian of softmax
        w = self.weights  # (2,)
        J = torch.diag(w) - w.unsqueeze(1) @ w.unsqueeze(0)  # shape (2,2)
        # Step 3: Full update with the Jacobian
        delta = J @ v
        self.xi = self.xi - self.beta * (delta + self.gamma * self.xi)
        self.weights = torch.softmax(self.xi, dim=0)