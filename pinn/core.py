"""Shared PINN building blocks: MLP, autograd derivatives, Adam -> L-BFGS training."""
import time
import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)

# Use the GPU when available (falls back to CPU). Setting the default device makes every tensor,
# nn.Parameter and nn.Linear created after this point live on that device.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
torch.set_default_device(DEVICE)


class MLP(nn.Module):
    """Fully connected network with tanh hidden activations and Glorot init."""

    def __init__(self, n_in, n_out, width, depth, init="normal"):
        super().__init__()
        dims = [n_in] + [width] * depth + [n_out]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        for layer in self.layers:
            if init == "normal":
                nn.init.xavier_normal_(layer.weight)
            else:
                nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        return self.layers[-1](x)


def grad(outputs, inputs):
    """d(outputs)/d(inputs) via automatic differentiation, keeping the graph so
    that higher-order derivatives (PDE residuals) can be taken."""
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True
    )[0]


def train(loss_fn, params, adam_epochs, lr, lbfgs_iters=0, log_every=500, callback=None, pre_step=None):
    """Adam for `adam_epochs`, then (optionally) L-BFGS until convergence.

    loss_fn() must return (total_loss, dict_of_components).
    Returns history: list of dicts with iteration + loss components.
    """
    history = []
    opt = torch.optim.Adam(params, lr=lr)
    t0 = time.time()
    for it in range(adam_epochs + 1):
        if pre_step:
            pre_step()   # e.g. draw fresh collocation points (Adam only; L-BFGS needs a fixed set)
        opt.zero_grad()
        loss, parts = loss_fn()
        loss.backward()
        opt.step()
        if it % log_every == 0:
            rec = {"iter": it, "loss": loss.item(), **{k: float(v) for k, v in parts.items()}}
            if callback:
                rec.update(callback())
            history.append(rec)
            print(f"[Adam {it:6d}] loss={loss.item():.3e}  ({time.time()-t0:.0f}s)", flush=True)

    if lbfgs_iters > 0:
        lb = torch.optim.LBFGS(
            params, lr=1.0, max_iter=lbfgs_iters, max_eval=int(lbfgs_iters * 1.25),
            history_size=50, tolerance_grad=1e-12, tolerance_change=1e-14,
            line_search_fn="strong_wolfe",
        )

        def closure():
            lb.zero_grad()
            l, _ = loss_fn()
            l.backward()
            return l

        lb.step(closure)
        loss, parts = loss_fn()
        rec = {"iter": adam_epochs + lbfgs_iters, "loss": loss.item(), **{k: float(v) for k, v in parts.items()}}
        if callback:
            rec.update(callback())
        history.append(rec)
        print(f"[L-BFGS done] loss={loss.item():.3e}  ({time.time()-t0:.0f}s)", flush=True)
    return history


def rel_l2(pred, true):
    return float(np.linalg.norm(pred - true) / np.linalg.norm(true))
