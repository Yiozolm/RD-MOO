# Balanced Rate-Distortion Optimization in Learned Image Compression

Unofficial Pytorch implementation of **CVPR2025** paper **Balanced Rate-Distortion Optimization in Learned Image Compression**.

Code offered by ChatGPT4.1, haven't tested or modified yet.

---

### How to use

``` python
# model = ... (your compression model)
# optimizer = BalancedRDTrajectoryOptimizer(model.parameters(), lr=1e-4)
# or
# optimizer = BalancedRDQPSolverOptimizer(model.parameters(), lr=1e-4)

for data in dataloader:
    # ... prepare input, forward pass, compute rate and distortion losses ...
    # loss_rate = ...
    # loss_dist = ...
    optimizer.step(loss_rate, loss_dist)

```