<div align="center">

# Diverse Low-Temperature Diffusion Sampling through Replica Exchange

[Zoe Wu](https://zoewu.com) · [Haldun Balim](https://haldunbalim.github.io/) · [Yilun Du](https://yilundu.github.io/)

[Paper](#) · [Project page](https://zoewu.com/STEER) · [BibTeX](#citation)

<img src="assets/key_video.gif" alt="STEER overview" width="100%">

</div>

STEER augments temporal score rescaling (TSR) with replica exchange. A ladder of
temperatures is simulated in parallel; each replica denoises with the noise
prediction rescaled for its own temperature, and neighbouring temperatures
periodically exchange through a Metropolis test. Samples are read out from the
target rung.

The method is training-free and works with any pretrained diffusion or flow model
that exposes a noise (or velocity) prediction.

## Repository layout

| Path | Contents |
| --- | --- |
| `steer/` | The algorithm: temperature ladder, rescaling, swap, readout |
| `tests/` | Unit tests for `steer/` |
| `toy/` | Self-contained 2D Gaussian mixture notebook comparing STEER with TSR, CNS, MCMC and CREPE |
| `diffusers/` | Text-to-image with Stable Diffusion 3 (COCO prompts, FID, precision/recall) |
| `pointmaze/` | Long-horizon planning on OGBench PointMaze |
| `robomimic/` | Diffusion Policy on robomimic manipulation tasks |
| `foldingdiff/` | Protein backbone generation with FoldingDiff |

Each experiment folder has its own README with setup and the commands used for
the paper. The `pointmaze/`, `robomimic/` and `foldingdiff/` folders contain only
the files we changed relative to the upstream projects; `setup.sh` in each folder
clones the upstream code at a pinned commit and applies them.

## Installation

```bash
git clone <this repository> STEER
cd STEER
pip install -e .
```

The only dependency of the core package is PyTorch. Experiment folders list their
own requirements.

## Algorithm

Let `lam_0 < ... < lam_{R-1}` be the temperature ladder and `alpha_t` the signal
coefficient of the sampler at step `t`. TSR replaces the noise prediction of a
replica at temperature `lam` by

```
eps_lam = c_t(lam) * eps,    c_t(lam) = 1 / (alpha_t * lam + 1 - alpha_t)
```

At every step STEER proposes swaps between rungs `(r, r + 1)`, alternating even
and odd pairs. With `x_r`, `s_r` the state and score of the replica holding rung
`r`, the swap is accepted with probability

```
min(1, exp(-1/2 * (c_t(lam_r) - c_t(lam_{r+1})) * <s_r + s_{r+1}, x_r - x_{r+1}>))
```

Swaps exchange temperature labels rather than data, so each replica keeps a
continuous trajectory.

```
x_T ~ N(0, I) for every replica;  temp_idx = identity
for t = T, ..., 1:
    eps = model(x_t, t)
    temp_idx = swap(x_t, score(eps), alpha_t, ladder, temp_idx, t)
    eps = rescale(eps, alpha_t, ladder, temp_idx)
    x_{t-1} = sampler_step(x_t, eps)
return x_0 at rung 0
```

## Usage

```python
from steer import init_temp_idx, lambda_ladder, rescale, select_rung, swap

ladder = lambda_ladder(lam_start, lam_end, n_replicas, device=device)
temp_idx = init_temp_idx(n_replicas, n_chains, device=device)
x = torch.randn(n_replicas * n_chains, *shape, device=device)

for step, t in enumerate(timesteps):
    eps = model(x, t)
    score = -eps / (1 - alpha_bar[t]).sqrt()
    swap(x, score, alpha_bar[t], ladder, temp_idx, step, generator=generator)
    eps = rescale(eps, alpha_bar[t], ladder, temp_idx)
    x = scheduler_step(x, eps, t)

samples = select_rung(x, temp_idx, rung=0)
```

Replica tensors have a leading dimension of `n_replicas * n_chains * k`, laid out
replica-major. `n_chains` independent ladders run side by side, and the `k` rows
inside a block share one swap decision (for example, all tokens of one image).
`temp_idx[r, c]` is the rung held by slot `r` of chain `c`.

| Function | Description |
| --- | --- |
| `lambda_ladder(lam_start, lam_end, n_replicas, spacing="linear")` | Temperature ladder (`linear`, `geometric`, `geometric_gamma`) |
| `init_temp_idx(n_replicas, n_chains=1)` | Identity assignment of rungs to slots |
| `rescale(eps, alpha, ladder, temp_idx)` | TSR-rescaled noise prediction for every slot |
| `swap(x, score, alpha, ladder, temp_idx, step, generator=None)` | One round of neighbour swaps; updates `temp_idx` in place |
| `select_rung(x, temp_idx, rung=0)` | Samples currently at a given rung |
| `sort_by_rung(x, temp_idx)` | Reorders replicas so that block `r` holds rung `r` |

TSR is recovered by skipping `swap`; with `n_replicas=1` it runs a single fixed
temperature `lam_start`.

## Experiments

| Folder | Upstream | Samplers |
| --- | --- | --- |
| `toy/` | — | vanilla, TSR, CNS, MCMC, CREPE, STEER |
| `diffusers/` | [huggingface/diffusers](https://github.com/huggingface/diffusers) 0.37.1 | vanilla, TSR, STEER |
| `pointmaze/` | [Diffusion-inference-scaling](https://github.com/XiangchengZhang/Diffusion-inference-scaling) | vanilla, TSR, CNS, MCMC, CREPE, STEER |
| `robomimic/` | [robomimic](https://github.com/ARISE-Initiative/robomimic) v0.5 | vanilla, TSR, CNS, MCMC, CREPE, STEER |
| `foldingdiff/` | [microsoft/foldingdiff](https://github.com/microsoft/foldingdiff) | vanilla, TSR, CNS, MCMC, STEER |

All experiments share the flags `--sampler`, `--lam_start`, `--lam_end`,
`--n_replicas` and `--seed`.

## Tests

```bash
pip install pytest
pytest tests
```

## Citation

```bibtex
@article{wu2026steer,
  title   = {Diverse Low-Temperature Diffusion Sampling through Replica Exchange},
  author  = {Wu, Zoe and Balim, Haldun and Du, Yilun},
  year    = {2026}
}
```
