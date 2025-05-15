import torch
from torch.optim.optimizer import Optimizer, required
import torch.nn.functional as F

class BalancedRDQPSolverOptimizer(Optimizer):
    """Balanced R-D optimizer (Solution 2: Quadratic Programming, fine-tuning)."""
    def __init__(self, params, lr=1e-4):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.device = None

    @torch.no_grad()
    def step(self, loss_rate, loss_dist):
        """Performs a single optimization step.
        Args:
            loss_rate: scalar tensor, the rate loss
            loss_dist: scalar tensor, the distortion loss
        """
        if self.device is None:
            self.device = loss_rate.device
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
        # Flatten and normalize
        eps = 1e-8
        v1 = torch.cat([g.view(-1) / (loss_rate.item() + eps) for g in grads_rate])
        v2 = torch.cat([g.view(-1) / (loss_dist.item() + eps) for g in grads_dist])
        Q = torch.tensor([
            [torch.dot(v1, v1), torch.dot(v1, v2)],
            [torch.dot(v2, v1), torch.dot(v2, v2)]
        ], device=self.device)
        one = torch.ones(2, device=self.device)
        Q_inv = torch.inverse(Q + 1e-8 * torch.eye(2, device=self.device))
        lam = 1.0 / (one @ Q_inv @ one)
        w = lam * (Q_inv @ one)
        w = F.softmax(w, dim=0)
        # Combine
        dt = [w[0]*g1/(loss_rate.item() + eps) + w[1]*g2/(loss_dist.item() + eps)
              for g1, g2 in zip(grads_rate, grads_dist)]
        c_t = 1.0 / (w[0] / (loss_rate.item() + eps) + w[1] / (loss_dist.item() + eps))
        dt = [c_t * d for d in dt]
        # Update parameters
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                p.data -= group['lr'] * dt[idx]
                idx += 1
