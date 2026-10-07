"""Problem 3 (inverse): damped single-degree-of-freedom oscillator  m u'' + c u' + k u = 0,
m = u(0) = u'(0) = 1.  Damping c and stiffness k are *unknown* and discovered from observations
of u(t) by treating them as trainable parameters alongside the network weights.
True values: c = 0.4, k = 4.
"""
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .core import MLP, grad, train, rel_l2

C_TRUE, K_TRUE, T_END = 0.4, 4.0, 15.0


def analytic(t, c=C_TRUE, k=K_TRUE, m=1.0, u0=1.0, v0=1.0):
    wn = np.sqrt(k / m); xi = c / (2 * m * wn); wd = wn * np.sqrt(1 - xi ** 2)
    return np.exp(-xi * wn * t) * (u0 * np.cos(wd * t) + (v0 + xi * wn * u0) / wd * np.sin(wd * t))


def run(adam_epochs, lbfgs_iters=0, lr=5e-4, seed=0, out="results", log_every=500, noise=0.0, tag="",
        n_data=50, n_scale=1.0, resample=False):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MLP(1, 1, width=32, depth=3, init="uniform")
    c = nn.Parameter(torch.tensor(0.1)); k = nn.Parameter(torch.tensor(2.0))   # initial guesses
    c.data = c.data.double(); k.data = k.data.double()

    td = torch.linspace(0, T_END, n_data).reshape(-1, 1)   # synthetic observations (paper: 50)
    ud = torch.tensor(analytic(td.numpy()))
    ud = ud + noise * ud.std() * torch.randn_like(ud)   # optional measurement noise
    nf = int(100 * n_scale)
    pts = {"tf": torch.rand(nf, 1) * T_END}                # collocation points (paper: 100)
    tt = torch.rand(10000, 1) * T_END
    t0 = torch.zeros(1, 1, requires_grad=True)
    wf = wb = wd = 0.25

    def residual(t):
        t = t.clone().requires_grad_(True)
        u = model(t)
        ut = grad(u, t)
        utt = grad(ut, t)
        return utt + c * ut + k * u

    def loss_fn():
        Lf = (residual(pts["tf"]) ** 2).mean()
        u0 = model(t0)
        Lb = ((u0 - 1.0) ** 2).mean() + ((grad(u0, t0) - 1.0) ** 2).mean()   # u(0)=1, u'(0)=1
        Ld = ((model(td) - ud) ** 2).mean()
        return wf * Lf + wb * Lb + wd * Ld, {"ode": Lf.item(), "ic": Lb.item(), "data": Ld.item()}

    def callback():
        return {"test_ode": (residual(tt) ** 2).mean().item(), "c": c.item(), "k": k.item()}

    hist = train(loss_fn, list(model.parameters()) + [c, k], adam_epochs, lr, lbfgs_iters, log_every, callback,
                 pre_step=(lambda: pts.update(tf=torch.rand(nf, 1) * T_END)) if resample else None)
    tg = np.linspace(0, T_END, 600)
    with torch.no_grad():
        pred = model(torch.tensor(tg).reshape(-1, 1)).numpy().ravel()
    true = analytic(tg)
    metrics = {"problem": "oscillator" + tag, "noise": noise, "c_est": c.item(), "k_est": k.item(),
               "c_true": C_TRUE, "k_true": K_TRUE,
               "c_rel_err_pct": abs(c.item() - C_TRUE) / C_TRUE * 100,
               "k_rel_err_pct": abs(k.item() - K_TRUE) / K_TRUE * 100,
               "solution_rel_l2": rel_l2(pred, true), "adam_epochs": adam_epochs, "lbfgs_iters": lbfgs_iters,
               "n_data": n_data, "n_f": nf, "n_scale": n_scale, "resample": resample}
    json.dump({"metrics": metrics, "history": hist}, open(f"{out}/oscillator{tag}.json", "w"), indent=1)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(tg, true, "k-", label="true solution"); ax[0].plot(tg, pred, "r--", label="PINN prediction")
    ax[0].plot(td.numpy(), ud.numpy(), "bo", ms=3, label="observations"); ax[0].set_xlabel("t"); ax[0].set_ylabel("u"); ax[0].legend(); ax[0].grid(alpha=.3)
    it = [h["iter"] for h in hist]
    ax[1].plot(it, [h["k"] for h in hist], "r", label="identified k"); ax[1].axhline(K_TRUE, color="r", ls=":", label="true k")
    ax[1].plot(it, [h["c"] for h in hist], "b", label="identified c"); ax[1].axhline(C_TRUE, color="b", ls=":", label="true c")
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel("parameter value"); ax[1].legend(); ax[1].grid(alpha=.3)
    ax[2].semilogy(it, [h["ode"] for h in hist], label="ODE train"); ax[2].semilogy(it, [h["test_ode"] for h in hist], label="ODE test")
    ax[2].semilogy(it, [h["data"] for h in hist], label="data MSE"); ax[2].set_xlabel("iteration"); ax[2].set_ylabel("loss"); ax[2].legend(); ax[2].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{out}/oscillator{tag}_results.png", dpi=150); plt.close()
    print("oscillator:", metrics)
    return metrics
