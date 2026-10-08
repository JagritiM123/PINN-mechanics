# Data-Driven Solutions and Discoveries in Mechanics using Physics-Informed Neural Networks (PINNs)
 
UE24CS352A – Machine Learning · Mini-Project
 
## Team
 
| Name | SRN |
|------|-----|
| Jagriti Mohan | PES2UG24CS200 |
| Jaladi Suhas  | PES2UG24CS201 |
 
Course: UE24CS352A Machine Learning, PES University · Faculty/TAs have been added as repository collaborators.
 
A from-scratch PyTorch implementation of the PINN framework (Raissi et al., 2019) applied to three
mechanics problems, following the Stanford CS229 report *"Data Driven Solutions and Discoveries in
Mechanics Using Physics Informed Neural Network"* (Zhang, Chen, Yang).
 
| # | Problem | Type | Governing equation | Reference solution |
|---|---------|------|--------------------|--------------------|
| 1 | 1-D soil consolidation (Terzaghi) | Forward (data-driven *solution*) | `p_t = p_xx` | Analytical Fourier series |
| 2 | Steady-state 2-D heat conduction | Forward | `T_xx + T_yy + 1 = 0` | Finite-difference solver |
| 3 | Damped spring-mass oscillator | Inverse (data-driven *discovery* of `c`, `k`) | `u'' + c u' + k u = 0` | Analytical solution |
 
**Extensions beyond the paper:** (a) a hard-constraint variant for Problem 2 using a smooth distance
function (the paper's suggested future work); (b) a noisy-data experiment for Problem 3;
(c) a finite-difference solver replaces deal.II as the numerical reference.
 
## Project structure
 
```
main.py                 CLI entry point
pinn/core.py            MLP, autograd derivative helper, Adam -> L-BFGS trainer
pinn/consolidation.py   Problem 1
pinn/heat.py            Problem 2 (soft vs hard boundary conditions)
pinn/oscillator.py      Problem 3 (parameter identification)
results/                Figures (*.png), metrics and loss histories (*.json), logs
docs/                   Write-up PDF and presentation
```
 
## Setup
 
Python 3.9+ (tested on 3.12), CPU is enough.

## Run
 
```bash
python main.py --problem all --budget default          # everything (~35 min on 1 CPU core)
python main.py --problem oscillator --budget quick     # ~1 min smoke test / live demo
python main.py --problem consolidation --budget quick  # ~1 min
python main.py --problem heat --budget quick           # ~3 min
python main.py --problem oscillator --budget full --noise 0.05   # inverse problem with 5% noisy data
```
 
## Generating more (synthetic) training data
 
A PINN has no labelled dataset: its training points are random coordinates where the PDE residual is evaluated, so they can be
generated without limit. Only the inverse problem uses observations, which are synthesised from the analytical solution.
 
## Method summary
 
A tanh MLP `u_hat(x; theta)` approximates the solution. Derivatives come from automatic
differentiation, and training minimises
 
`L = w_f * L_pde + w_b * L_bc/ic + w_d * L_data`
 
over collocation points sampled in the domain. For inverse problems the unknown physical parameters
are trainable variables optimised together with the weights. Optimisation uses Adam followed by L-BFGS.
Problem 1 is non-dimensionalised so all variables are O(1) and the Dirichlet condition is hard-wired by
multiplying the network output by `x`.
 
## Reproducibility
 
Fixed seeds (`--seed 0` default) and float64 arithmetic; collocation sets are fixed unless `--resample` is used.
Results depend on the seed (especially consolidation) and vary across PyTorch versions and hardware, so compare
several seeds rather than a single run.
