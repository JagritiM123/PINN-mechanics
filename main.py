"""Command-line entry point.

    python main.py --problem all --budget default
    python main.py --problem oscillator --budget quick      # ~1 minute smoke test / live demo
    python main.py --problem consolidation --budget full    # paper-scale training
"""
import argparse
import json
import os
from pinn import consolidation, heat, oscillator

# (adam_epochs, lbfgs_iters) per problem.  'full' mirrors the paper's settings.
BUDGETS = {
    "quick":   {"consolidation": (1500, 300),   "heat": (1500, 300),   "oscillator": (6000, 2000)},
    "default": {"consolidation": (10000, 2000), "heat": (5000, 1000), "oscillator": (20000, 20000)},
    "full":    {"consolidation": (50000, 10000), "heat": (50000, 10000), "oscillator": (100000, 20000)},
}


def main():
    ap = argparse.ArgumentParser(description="PINNs for mechanics: forward and inverse problems")
    ap.add_argument("--problem", choices=["consolidation", "heat", "oscillator", "all"], default="all")
    ap.add_argument("--budget", choices=list(BUDGETS), default="default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.0, help="relative noise on oscillator observations")
    ap.add_argument("--n_scale", type=float, default=1.0, help="multiply collocation-point counts (1.0 = paper)")
    ap.add_argument("--n_data", type=int, default=50, help="number of synthetic oscillator observations (paper: 50)")
    ap.add_argument("--resample", action="store_true", help="draw fresh collocation points every Adam step")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    B = BUDGETS[a.budget]
    summary = {}
    if a.problem in ("consolidation", "all"):
        summary["consolidation"] = consolidation.run(*B["consolidation"], seed=a.seed, out=a.out, n_scale=a.n_scale, resample=a.resample)
    if a.problem in ("heat", "all"):
        summary["heat"] = heat.run(*B["heat"], seed=a.seed, out=a.out, n_scale=a.n_scale, resample=a.resample)
    if a.problem in ("oscillator", "all"):
        summary["oscillator"] = oscillator.run(B["oscillator"][0], B["oscillator"][1], seed=a.seed, out=a.out,
                                               noise=a.noise, tag="" if a.noise == 0 else f"_noise{int(a.noise*100)}",
                                               n_data=a.n_data, n_scale=a.n_scale, resample=a.resample)
    suffix = (f"_noise{int(a.noise*100)}" if a.noise else "") + (f"_x{a.n_scale:g}" if a.n_scale != 1 else "") \
             + (f"_n{a.n_data}" if a.n_data != 50 else "") + ("_resample" if a.resample else "")
    json.dump(summary, open(os.path.join(a.out, f"summary_{a.problem}_{a.budget}{suffix}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
