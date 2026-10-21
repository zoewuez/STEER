# 2D Gaussian mixtures

`steer_2d.ipynb` is a self-contained comparison of STEER with TSR, CNS, MCMC and CREPE on synthetic
2D mixtures. It trains two small DDPMs from scratch and samples the tempered target
p(x)<sup>1/λ</sup>, using exact grid samples from that target as the reference.

| Benchmark | Distribution | λ | STEER ladder | Metrics |
|---|---|---|---|---|
| 1 | 19 clustered modes + 1 rare, narrow, isolated mode | 0.9 | 0.9 → 20.0, 4 replicas | W2, MMD², mode coverage, mode-weight TV, occupancy entropy, hits on the isolated mode |
| 2 | 40 equal-weight, well-separated modes | 0.9 | 0.9 → 4.0, 3 replicas | W2, MMD², mode coverage, mode-weight TV, occupancy entropy |

Every method is run for five seeds with 1000 samples per seed. Each benchmark reports a metrics
table, a bar chart of every metric with the exact-sample noise floor as a dashed line, and sample
density histograms pooled over seeds.

## Running

Install the `steer` package from the repository root, plus the notebook dependencies:

```bash
pip install -e ..
pip install numpy scipy matplotlib pandas pot jupyter
jupyter notebook steer_2d.ipynb
```

If `steer` is not installed, the notebook adds the repository root to `sys.path` itself, as long
as it is opened from this directory.

On Colab, clone the repository, run `%pip install -e STEER pot`, and open
`STEER/toy/steer_2d.ipynb`.

The notebook runs on CPU in about 10 minutes (two cores); most of that time is training the two denoisers.
To use a GPU, set `DEVICE` in the setup cell. Sample values then differ slightly from the CPU run
because the random streams differ.

## Where STEER is used

The samplers in the notebook call the shared `steer` package and nothing else for the replica
exchange:

```python
temp_idx = init_temp_idx(n_replicas, num_samples)
lam_ladder = lambda_ladder(lam_start, lam_end, n_replicas)
for t in reversed(range(T)):
    eps = model(x, t)
    score = -eps / sqrt(clamp(1 - alpha_bar[t], min=1e-3))
    swap(x, score, alpha_bar[t], lam_ladder, temp_idx, t, generator)
    eps = rescale(eps, alpha_bar[t], lam_ladder, temp_idx)
    x = ddpm_step(x, eps, t)
sample = select_rung(x, temp_idx, 0)
```
