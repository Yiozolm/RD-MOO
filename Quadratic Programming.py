import torch
import torch.nn.functional as F

class BalancedRDOptimizerSolution2:
    def __init__(self, model, lr=1e-4):
        self.model = model
        self.lr = lr
        self.device = next(model.parameters()).device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def step(self, loss_rate, loss_dist):
        # Compute gradients for log-loss
        self.optimizer.zero_grad()
        loss_rate.backward(retain_graph=True)
        grad_rate = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in self.model.parameters()]
        self.optimizer.zero_grad()
        loss_dist.backward(retain_graph=True)
        grad_dist = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in self.model.parameters()]
        self.optimizer.zero_grad()

        # Compute normalized gradients
        norm_grad_rate = torch.cat([g.view(-1) / (loss_rate.item() + 1e-8) for g in grad_rate])
        norm_grad_dist = torch.cat([g.view(-1) / (loss_dist.item() + 1e-8) for g in grad_dist])

        # Compute Hessian matrix Q (2x2)
        Q = torch.tensor([
            [torch.dot(norm_grad_rate, norm_grad_rate), torch.dot(norm_grad_rate, norm_grad_dist)],
            [torch.dot(norm_grad_dist, norm_grad_rate), torch.dot(norm_grad_dist, norm_grad_dist)]
        ], device=self.device)

        one_vec = torch.ones(2, device=self.device)
        Q_inv = torch.inverse(Q + 1e-8*torch.eye(2, device=self.device))
        lam = 1.0 / (one_vec @ Q_inv @ one_vec)
        w = lam * (Q_inv @ one_vec)
        w = F.softmax(w, dim=0)  # project to simplex for stability

        # Compute final weighted gradient
        # Expand to per-parameter shape:
        dt = [w[0]*g1/(loss_rate.item() + 1e-8) + w[1]*g2/(loss_dist.item() + 1e-8)
              for g1, g2 in zip(grad_rate, grad_dist)]
        c_t = 1.0 / (w[0] / (loss_rate.item() + 1e-8) + w[1] / (loss_dist.item() + 1e-8))
        dt = [c_t * d for d in dt]

        # Manual parameter update
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), dt):
                p -= self.lr * d

