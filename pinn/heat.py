"""Problem 2 (forward): steady-state 2-D heat conduction (Poisson), T_xx + T_yy + 1 = 0 on [-1,1]^2,
T = 0 on the boundary.  Reference: second-order finite-difference solution (fine grid).

Two PINN variants are compared:
  * soft : plain network, Dirichlet BC enforced through a loss term (as in the paper)
  * hard : T = d(x,y) * N(x,y) with d = (1-x^2)(1-y^2), so the BC holds exactly
           (the improvement the paper proposes as future work, via a smooth distance function)
"""
import json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .core import MLP, grad, train, rel_l2


def fd_reference(n=201):
    """Solve -lap(T) = 1 on [-1,1]^2 with T=0 on the boundary; returns grid on n x n points."""
    h = 2.0 / (n - 1)
    m = n - 2
    I = sp.identity(m)
    D = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(m, m))
    A = (sp.kron(I, D) + sp.kron(D, I)) / h ** 2
    T = np.zeros((n, n))
    T[1:-1, 1:-1] = spla.spsolve(A.tocsc(), np.ones(m * m)).reshape(m, m)
    x = np.linspace(-1, 1, n)
    return x, T


def run(adam_epochs, lbfgs_iters, lr=5e-4, seed=0, out="results", log_every=500, n_scale=1.0, resample=False):
    xr, Tr = fd_reference(201)
    xs = xr[::2]; ref = Tr[::2, ::2]           # 101 x 101 evaluation grid
    XG, YG = np.meshgrid(xs, xs, indexing="ij")
    G = torch.tensor(np.stack([XG.ravel(), YG.ravel()], 1))

    results, histories, preds = {}, {}, {}
    for variant in ["soft", "hard"]:
        torch.manual_seed(seed)
        model = MLP(2, 1, width=50, depth=4, init="uniform")

        def T_fn(xy):
            out_ = model(xy)
            if variant == "hard":
                out_ = (1 - xy[:, :1] ** 2) * (1 - xy[:, 1:] ** 2) * out_
            return out_

        def residual(xy):
            xy = xy.clone().requires_grad_(True)
            T = T_fn(xy)
            g = grad(T, xy)
            Txx = grad(g[:, :1], xy)[:, :1]
            Tyy = grad(g[:, 1:], xy)[:, 1:]
            return Txx + Tyy + 1.0

        nf, ns = int(1500 * n_scale), int(125 * n_scale)

        def sample():   # synthetic training points (paper: 1500 interior / 500 boundary at n_scale=1)
            s = torch.rand(ns, 1) * 2 - 1
            one = torch.ones_like(s)
            xb = torch.cat([torch.cat([s, -one], 1), torch.cat([s, one], 1),
                            torch.cat([-one, s], 1), torch.cat([one, s], 1)], 0)
            return {"xf": torch.rand(nf, 2) * 2 - 1, "xb": xb}
        pts = sample()
        xt = torch.rand(10000, 2) * 2 - 1
        wf = wb = 0.5

        def loss_fn():
            xf, xb = pts["xf"], pts["xb"]
            Lf = (residual(xf) ** 2).mean()
            Lb = (T_fn(xb) ** 2).mean()
            return wf * Lf + wb * Lb, {"pde": Lf.item(), "bc": Lb.item()}

        def predict():
            with torch.no_grad():
                return T_fn(G).cpu().numpy().reshape(XG.shape)

        def callback():
            return {"test_pde": (residual(xt) ** 2).mean().item(), "rel_l2": rel_l2(predict(), ref)}

        print(f"--- heat variant: {variant}")
        hist = train(loss_fn, list(model.parameters()), adam_epochs, lr, lbfgs_iters, log_every, callback,
                     pre_step=(lambda: pts.update(sample())) if resample else None)
        pred = predict()
        err = np.abs(pred - ref)
        bd = np.concatenate([pred[0], pred[-1], pred[:, 0], pred[:, -1]])
        results[variant] = {"rel_l2": rel_l2(pred, ref), "max_abs_err": float(err.max()),
                            "mean_abs_err": float(err.mean()), "max_abs_boundary_T": float(np.abs(bd).max())}
        histories[variant], preds[variant] = hist, pred

    metrics = {"problem": "heat", "adam_epochs": adam_epochs, "lbfgs_iters": lbfgs_iters, "n_scale": n_scale, "resample": resample,
               "fd_center_T": float(ref[50, 50]), **{f"{k}_{m}": v for k, r in results.items() for m, v in r.items()}}
    json.dump({"metrics": metrics, "history": histories}, open(f"{out}/heat.json", "w"), indent=1)

    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    ext = [-1, 1, -1, 1]
    im = ax[0, 0].imshow(ref.T, origin="lower", extent=ext, cmap="jet"); ax[0, 0].set_title("Reference (finite difference)"); plt.colorbar(im, ax=ax[0, 0])
    im = ax[0, 1].imshow(preds["soft"].T, origin="lower", extent=ext, cmap="jet"); ax[0, 1].set_title("PINN (soft BC, as in paper)"); plt.colorbar(im, ax=ax[0, 1])
    im = ax[0, 2].imshow(preds["hard"].T, origin="lower", extent=ext, cmap="jet"); ax[0, 2].set_title("PINN (hard BC, distance function)"); plt.colorbar(im, ax=ax[0, 2])
    vmax = max(np.abs(preds["soft"] - ref).max(), np.abs(preds["hard"] - ref).max())
    im = ax[1, 0].imshow(np.abs(preds["soft"] - ref).T, origin="lower", extent=ext, cmap="jet", vmin=0, vmax=vmax); ax[1, 0].set_title("|error| soft BC"); plt.colorbar(im, ax=ax[1, 0])
    im = ax[1, 1].imshow(np.abs(preds["hard"] - ref).T, origin="lower", extent=ext, cmap="jet", vmin=0, vmax=vmax); ax[1, 1].set_title("|error| hard BC"); plt.colorbar(im, ax=ax[1, 1])
    for v, ls in [("soft", "-"), ("hard", "--")]:
        it = [h["iter"] for h in histories[v]]
        ax[1, 2].semilogy(it, [h["pde"] for h in histories[v]], ls, label=f"{v}: PDE train")
        ax[1, 2].semilogy(it, [h["test_pde"] for h in histories[v]], ls, alpha=.6, label=f"{v}: PDE test")
    ax[1, 2].set_xlabel("iteration"); ax[1, 2].set_ylabel("loss"); ax[1, 2].legend(fontsize=7); ax[1, 2].grid(alpha=.3)
    for a in ax.ravel()[:5]:
        a.set_xlabel("x"); a.set_ylabel("y")
    plt.tight_layout(); plt.savefig(f"{out}/heat_results.png", dpi=150); plt.close()
    print("heat:", metrics)
    return metrics
