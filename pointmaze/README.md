# STEER on PointMaze

Long-horizon planning in the OGBench PointMaze environments with a pretrained
trajectory diffusion model. STEER runs inside the inference-time search methods of
[Diffusion-inference-scaling](https://github.com/XiangchengZhang/Diffusion-inference-scaling)
(best-of-N, BFS, DFS): the `n_replicas` particles of a best-of-N search form one
temperature ladder, each particle denoises with the score rescaled for its rung, and
neighbouring rungs exchange temperatures at every step. The verifier then picks the
best particle as usual.

This folder only contains the files we changed. `setup.sh` checks out the upstream
repository at the commit we built on and copies them in.

```
overlay/                     files copied onto upstream pointmaze/
  run.py                     CLI: --sampler, --lam_start, --lam_end, --n_replicas, --seed
  search/configs.py          sampler, ladder, MCMC and CREPE settings
  search/script_utils.py     best-of-N runs with --n_replicas particles; CREPE policy
  search/search_policy.py    builds temp_idx for each sampling call
  search/methods/tfg.py      swap + rescale (STEER, TSR), Langevin corrector (MCMC)
  search/methods/base_guidance.py   scaled ancestral noise (CNS)
  search/methods/bfs.py, dfs.py     pass temp_idx to the local search step
  search/methods/crepe.py    CREPE baseline
scripts/sweep.sh             seeds x samplers x ladder sizes
setup.sh
```

## Installation

```bash
./setup.sh                               # clones into ./external/Diffusion-inference-scaling
cd external/Diffusion-inference-scaling/pointmaze

conda create -n maze python=3.10 && conda activate maze
pip install -r requirements.txt
pip install -e ogbench
pip install -e .
pip install -e ../../../..               # the steer package at the repository root
```

Download and extract the pretrained models from
[here](https://drive.google.com/file/d/1ZMhoOkLLMozUdADKph3oed_OwzAYtiD1/view?usp=sharing)
into `logs/`. For the Ultra maze, download the trajectory dataset from
[here](https://drive.google.com/file/d/1doAvARCm04axeXFUn8ETRgWfcoMt84m3/view?usp=sharing)
and place `pointmaze-ultra-navigate-v0.npz` and `pointmaze-ultra-navigate-v0-val.npz`
in `~/.ogbench/data`.

On headless machines set `MUJOCO_GL=egl`, or `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`
if EGL is unavailable.

## Usage

Best-of-4 with a STEER ladder from 0.9 to 0.94:

```bash
python run.py --dataset pointmaze-giant-navigate-v0 --method bon \
    --sampler steer --lam_start 0.9 --lam_end 0.94 --n_replicas 4 --seed 1
```

The same search with the baselines:

```bash
python run.py --dataset pointmaze-giant-navigate-v0 --method bon --sampler tsr   --lam_start 0.9 --lam_end 0.94 --n_replicas 4
python run.py --dataset pointmaze-giant-navigate-v0 --method bon --sampler cns   --lam_start 0.9 --n_replicas 4
python run.py --dataset pointmaze-giant-navigate-v0 --method bon --sampler mcmc  --lam_start 0.9 --n_replicas 4
python run.py --dataset pointmaze-giant-navigate-v0 --method bon --sampler crepe --n_replicas 4
python run.py --dataset pointmaze-giant-navigate-v0 --method dfs --sampler vanilla
```

The full sweep (5 seeds; ladders of 2, 4, 8 and 16 replicas with
`lam_end = lam_start + 0.01 * n_replicas`; BFS and DFS references):

```bash
bash sweep.sh
SAMPLERS="steer" N_REPLICAS="4 8" SEEDS="1" bash sweep.sh    # a subset
```

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `pointmaze-ultra-navigate-v0` | `pointmaze-giant-navigate-v0` or `pointmaze-ultra-navigate-v0` |
| `--method` | `bon` | Search method: `bon`, `bfs-resampling`, `bfs-pruning`, `dfs` |
| `--sampler` | `vanilla` | `vanilla`, `tsr`, `steer`, `cns`, `mcmc`, `crepe` |
| `--lam_start` | `1.0` | First rung of the ladder; the scale used by `cns` and `mcmc` |
| `--lam_end` | `1.0` | Last rung of the ladder |
| `--n_replicas` | `4` | Ladder size; also the number of best-of-N particles on the Giant maze |
| `--seed` | `42` | Random seed |
| `--device` | `cuda` | Torch device |

`tsr` gives each particle a fixed rung of the ladder, `steer` additionally swaps
rungs between neighbours, `cns` scales the injected noise by `sqrt(lam_start)`,
`mcmc` adds a Langevin corrector at each noise level, and `crepe` refines the
best-of-N samples with reward-tilted parallel tempering over diffusion time. Extra
MCMC and CREPE settings are in `search/configs.py`.

## Output

Each configuration appends one line to `results_<method>_<dataset>.txt`:

```
Maze: pointmaze-giant-navigate-v0 | Sampler: steer | Lam 0.9 to 0.94 with 4 | Method: bon | Particles: 4 | Seed: 1 | Compute: 128.0 | Success Rate: 92.5
```

`Compute` is the average number of network evaluations per plan and `Success Rate`
is the percentage of the 40 evaluation episodes that reach the goal. Plans, rollouts
and per-episode results are written under `logs/<dataset>/inference/`.

`--seed` fixes the diffusion noise, but episode start and goal positions are drawn
from the global NumPy RNG and the guidance gradients use non-deterministic CUDA
kernels, so success rates vary slightly between runs. Report the mean over seeds.

## Acknowledgements

The search methods, maze verifier and pretrained models come from
[Diffusion-inference-scaling](https://github.com/XiangchengZhang/Diffusion-inference-scaling),
which builds on [diffuser](https://github.com/jannerm/diffuser) and
[OGBench](https://github.com/seohongpark/ogbench). Please cite their work alongside ours.
