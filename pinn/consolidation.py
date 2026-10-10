"""Problem 1 (forward): 1-D Terzaghi consolidation, dp/dt = d2p/dx2 in dimensionless form.

Domain x in (0,1), t in (0,t_max]; p(0,t)=0 (drained top), dp/dx(1,t)=0 (impermeable base),
p(x,0)=1 (initial pore pressure).  The network output is multiplied by x so the
Dirichlet condition at x=0 holds exactly ("hard" constraint, as in the paper);
the Neumann and initial conditions are enforced as soft loss terms.
"""
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .core import MLP, grad, train, rel_l2

T_MAX = 0.5


def analytic(x, t, terms=3000):
    """Series solution (odd m): p = sum 4/(m pi) sin(m pi x/2) exp(-m^2 pi^2 t/4)."""
    m = np.arange(1, 2 * terms, 2)[None, None, :]
    X, T = np.meshgrid(x, t, indexing="ij")
    X, T = X[..., None], T[..., None]
    return np.sum(4 / (m * np.pi) * np.sin(m * np.pi * X / 2) * np.exp(-(m ** 2) * np.pi ** 2 * T / 4), axis=-1)


def run(adam_epochs, lbfgs_iters, lr=1e-3, seed=0, out="results", log_every=500, n_scale=1.0, resample=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP(2, 1, width=50, depth=5, init="normal")

    def p_fn(x, t):
        return x * model(torch.cat([x, t], 1))  # hard-enforces p(0,t)=0

    def residual(x, t):
        x = x.clone().requires_grad_(True)
        t = t.clone().requires_grad_(True)
        p = p_fn(x, t)
        p_t = grad(p, t)
        p_x = grad(p, x)
        p_xx = grad(p_x, x)
        return p_t - p_xx

    rnd = lambda n, lo, hi: torch.rand(n, 1) * (hi - lo) + lo
    nf, nb, ni = int(500 * n_scale), int(250 * n_scale), int(125 * n_scale)

    def sample():   # synthetic training points (paper: 500 / 250 / 125 at n_scale=1)
        return {"xf": rnd(nf, 0, 1), "tf": rnd(nf, 0, T_MAX),   # PDE residual points (Gamma_f)
                "tb": rnd(nb, 0, T_MAX),                         # Neumann boundary x=1
                "xi": rnd(ni, 0, 1)}                             # initial condition t=0
    pts = sample()
    wf = wb = 0.25

    def loss_fn():
        xf, tf, tb, xi = pts["xf"], pts["tf"], pts["tb"], pts["xi"]
        Lf = (residual(xf, tf) ** 2).mean()
        xb = torch.ones_like(tb).requires_grad_(True)
        px = grad(p_fn(xb, tb), xb)
        Lb = (px ** 2).mean() + ((p_fn(xi, torch.zeros_like(xi)) - 1.0) ** 2).mean()
        return wf * Lf + wb * Lb, {"pde": Lf.item(), "bc_ic": Lb.item()}

    # fixed evaluation sets
    xs, ts = np.linspace(0, 1, 101), np.linspace(0.005, T_MAX, 100)
    ref = analytic(xs, ts)
    XG, TG = np.meshgrid(xs, ts, indexing="ij")
    Xg = torch.tensor(XG.reshape(-1, 1)); Tg = torch.tensor(TG.reshape(-1, 1))
    # test residual points; t >= 0.02 because the initial jump makes the residual ill-defined as t -> 0
    xt_, tt_ = rnd(10011, 0, 1), rnd(10011, 0.02, T_MAX)

    def predict():
        with torch.no_grad():
            return p_fn(Xg, Tg).cpu().numpy().reshape(XG.shape)

    def callback():
        test = (residual(xt_, tt_) ** 2).mean().item()
        return {"test_pde": test, "rel_l2": rel_l2(predict(), ref)}

    hist = train(loss_fn, list(model.parameters()), adam_epochs, lr, lbfgs_iters, log_every, callback,
                 pre_step=(lambda: pts.update(sample())) if resample else None)
    pred = predict()
    err = np.abs(pred - ref)
    metrics = {"problem": "consolidation", "rel_l2": rel_l2(pred, ref), "max_abs_err": float(err.max()),
               "mean_abs_err": float(err.mean()), "adam_epochs": adam_epochs, "lbfgs_iters": lbfgs_iters,
               "n_f": nf, "n_scale": n_scale, "resample": resample}
    json.dump({"metrics": metrics, "history": hist}, open(f"{out}/consolidation.json", "w"), indent=1)

    # ---- figures
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ext = [0, 1, ts[0], ts[-1]]
    for a, f, ttl in [(ax[0, 0], ref, "Analytical solution"), (ax[0, 1], pred, "PINN solution")]:
        im = a.imshow(f.T, origin="lower", extent=ext, aspect="auto", cmap="jet", vmin=0, vmax=1)
        a.set_title(ttl); a.set_xlabel(r"$\hat{x}$"); a.set_ylabel(r"$\hat{t}$"); plt.colorbar(im, ax=a)
    im = ax[1, 0].imshow(err.T, origin="lower", extent=ext, aspect="auto", cmap="jet")
    ax[1, 0].set_title("Absolute error"); ax[1, 0].set_xlabel(r"$\hat{x}$"); ax[1, 0].set_ylabel(r"$\hat{t}$")
    plt.colorbar(im, ax=ax[1, 0])
    it = [h["iter"] for h in hist]
    ax[1, 1].semilogy(it, [h["pde"] for h in hist], label="PDE train loss")
    ax[1, 1].semilogy(it, [h["test_pde"] for h in hist], label="PDE test loss")
    ax[1, 1].semilogy(it, [h["bc_ic"] for h in hist], label="BC+IC loss")
    ax[1, 1].set_xlabel("iteration"); ax[1, 1].set_ylabel("loss"); ax[1, 1].legend(); ax[1, 1].grid(alpha=.3)
    ax[1, 1].set_title("Convergence history")
    plt.tight_layout(); plt.savefig(f"{out}/consolidation_results.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 4.2))
    for tt in [0.01, 0.05, 0.1, 0.25, 0.5]:
        j = int(np.argmin(np.abs(ts - tt)))
        l, = plt.plot(xs, ref[:, j], "-", lw=2, alpha=.6)
        plt.plot(xs, pred[:, j], "--", color="k", lw=1)
        plt.text(0.62, ref[int(0.62*100), j] + 0.02, rf"$\hat t$={ts[j]:.2f}", color=l.get_color(), fontsize=8)
    plt.plot([], [], "-", color="gray", label="analytical"); plt.plot([], [], "k--", label="PINN")
    plt.xlabel(r"$\hat{x}$ (depth)"); plt.ylabel(r"$\hat{p}$ (pore pressure)"); plt.legend(); plt.grid(alpha=.3)
    plt.title("Pore-pressure dissipation profiles"); plt.tight_layout()
    plt.savefig(f"{out}/consolidation_profiles.png", dpi=150); plt.close()
    print("consolidation:", metrics)
    return metrics
