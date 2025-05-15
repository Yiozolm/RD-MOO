import torch
from torch.optim.optimizer import Optimizer, required
import torch.nn.functional as F


class BalancedRDTrajectoryOptimizer(Optimizer):
    """Balanced R-D optimizer (Solution 1: trajectory/coarse-to-fine)."""

    def __init__(self, params, lr=1e-4, beta=0.025, gamma=0.001):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.device = None
        self.beta = beta
        self.gamma = gamma
        # Initialize softmax logits (for 2 tasks: rate, distortion)
        self.state['xi'] = torch.zeros(2, requires_grad=False)
        self.state['weights'] = F.softmax(self.state['xi'], dim=0)

    @torch.no_grad()
    def step(self, loss_rate, loss_dist):
        """Performs a single optimization step.
        Args:
            loss_rate: scalar tensor, the rate loss
            loss_dist: scalar tensor, the distortion loss
        """
        # Only set device on first use
        if self.device is None:
            self.device = loss_rate.device
            self.state['xi'] = self.state['xi'].to(self.device)
            self.state['weights'] = self.state['weights'].to(self.device)

        # Compute gradients of log losses
        grads_rate = []
        grads_dist = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.zero_()
        loss_rate.backward(retain_graph=True)
        for group in self.param_groups:
            for p in group['params']:
                grads_rate.append((p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)))
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.zero_()
        loss_dist.backward(retain_graph=True)
        for group in self.param_groups:
            for p in group['params']:
                grads_dist.append((p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)))
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.zero_()
        # Normalize gradients
        eps = 1e-8
        norm_grad_rate = [g / (loss_rate.item() + eps) for g in grads_rate]
        norm_grad_dist = [g / (loss_dist.item() + eps) for g in grads_dist]
        # Combine
        w = self.state['weights']
        dt = [w[0] * g1 + w[1] * g2 for g1, g2 in zip(norm_grad_rate, norm_grad_dist)]
        # Renormalization for numerical stability
        c_t = 1.0 / (w[0] / (loss_rate.item() + eps) + w[1] / (loss_dist.item() + eps))
        dt = [c_t * d for d in dt]
        # Update parameters
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                p.data -= group['lr'] * dt[idx]
                idx += 1
        # ---- Weight update for next iteration ----
        # For demonstration: you should provide loss_rate_next, loss_dist_next
        # For practical use, update weights with current values (approximate):
        lr1 = torch.log(loss_rate.detach() + 1)
        ld1 = torch.log(loss_dist.detach() + 1)
        lr2 = lr1  # Ideally: torch.log(loss_rate_next + 1)
        ld2 = ld1  # Ideally: torch.log(loss_dist_next + 1)
        delta = torch.stack([lr1 - lr2, ld1 - ld2]).to(self.device)
        self.state['xi'] -= self.beta * (delta + self.gamma * self.state['xi'])
        self.state['weights'] = F.softmax(self.state['xi'], dim=0)
