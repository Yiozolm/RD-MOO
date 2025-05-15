import torch
import torch.nn.functional as F


class BalancedRDOptimizerSolution1:
    def __init__(self, model, lr=1e-4, beta=0.025, gamma=0.001):
        self.model = model
        self.lr = lr
        self.beta = beta
        self.gamma = gamma
        self.device = next(model.parameters()).device
        # Initialize softmax logits for weights (2 tasks: [rate, distortion])
        self.xi = torch.zeros(2, requires_grad=False, device=self.device)
        self.weights = F.softmax(self.xi, dim=0)
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
        norm_grad_rate = [g / (loss_rate.item() + 1e-8) for g in grad_rate]
        norm_grad_dist = [g / (loss_dist.item() + 1e-8) for g in grad_dist]

        # Compute convex combination
        dt = [self.weights[0] * g1 + self.weights[1] * g2 for g1, g2 in zip(norm_grad_rate, norm_grad_dist)]

        # Normalization constant for stability
        c_t = 1.0 / (self.weights[0] / (loss_rate.item() + 1e-8) + self.weights[1] / (loss_dist.item() + 1e-8))
        dt = [c_t * d for d in dt]

        # Manual parameter update
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), dt):
                p -= self.lr * d

        # Compute loss for next step
        # Forward pass with updated weights (pseudo, user must re-compute in main training loop)
        # loss_rate_next, loss_dist_next = ... # recompute after weight update

        # For demonstration, assume loss_rate_next and loss_dist_next are known
        # Here, just re-use current losses as placeholders:
        loss_rate_next, loss_dist_next = loss_rate, loss_dist

        # Softmax logit update (weight decay included)
        delta = torch.tensor([
            torch.log(loss_rate.item() + 1) - torch.log(loss_rate_next.item() + 1),
            torch.log(loss_dist.item() + 1) - torch.log(loss_dist_next.item() + 1)
        ], device=self.device)
        self.xi -= self.beta * (delta + self.gamma * self.xi)
        self.weights = F.softmax(self.xi, dim=0)

