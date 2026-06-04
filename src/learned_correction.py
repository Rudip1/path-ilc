"""
Learned correction layer: a small neural network that predicts an ILC
correction table from path geometry.

Motivation (the third goal, and the paper's open question):
  Classical ILC must run several trials on EACH new path. In production the
  part shifts slightly trial-to-trial, so the path is never quite the same,
  and re-running trials each time is costly. We ask: can a model look at a
  new path's geometry and predict a good correction immediately -- zero
  trials?

How it stays honest:
  - Training targets are ILC-converged tables, each obtained the hard way by
    actually running trials on the MuJoCo plant for that path. We never hand
    the network a closed-form error.
  - The TEST path is one the network never saw during training. The metric
    is the Cartesian error when the predicted correction is applied to the
    real plant on that unseen path -- it cannot be gamed by construction.
  - The honest framing in the README: this shows the *architecture* (learned
    layer on top of classical ILC, exactly like a context-adaptive policy on
    top of a classical planner). On a 3-DOF sim with smooth paths the mapping
    is easy; the real 6-DOF, load/temperature-dependent problem is the open
    research question.

This mirrors a context-adaptive learning layer on top of a classical MPPI
planner: same shape, different domain.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def path_features(lambdas, xz_ref):
    """Geometry descriptor the network sees: the desired (x, z) along the
    path, flattened. This is the 'context' -- analogous to the environment
    context in a context-adaptive controller."""
    return xz_ref.reshape(-1)


if _HAS_TORCH:
    class CorrectionNet(nn.Module):
        """Maps a path-geometry vector to a flattened (N x 3) correction table.

        Deliberately small: the point is the architecture, not raw capacity.
        """

        def __init__(self, in_dim, out_dim, hidden=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, out_dim),
            )

        def forward(self, x):
            return self.net(x)

    def train_correction_net(features, tables, epochs=400, lr=1e-3, seed=0):
        """Fit the network. 'features' is (P, F), 'tables' is (P, N*3).
        Returns the trained net and a (mean, std) normalizer for inputs."""
        torch.manual_seed(seed)
        X = torch.tensor(np.asarray(features), dtype=torch.float32)
        Y = torch.tensor(np.asarray(tables), dtype=torch.float32)
        mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True) + 1e-8
        Xn = (X - mu) / sd
        net = CorrectionNet(Xn.shape[1], Y.shape[1])
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for ep in range(epochs):
            opt.zero_grad()
            loss = loss_fn(net(Xn), Y)
            loss.backward()
            opt.step()
        return net, (mu, sd)

    def predict_table(net, norm, feature_vec, N, dof=3):
        """Predict a correction table for one path's feature vector.
        `dof` is the joint dimension (3 for the toy arm, 7 for the KUKA)."""
        mu, sd = norm
        x = torch.tensor(feature_vec[None, :], dtype=torch.float32)
        xn = (x - mu) / sd
        with torch.no_grad():
            y = net(xn).numpy().reshape(N, dof)
        return y
