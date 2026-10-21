# STEER on robomimic

STEER applied to [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) in
[robomimic](https://github.com/ARISE-Initiative/robomimic) on the Lift, Can, Square and Tool Hang
tasks. The policies are trained on a random subset of the proficient-human demonstrations, and
STEER is applied only at inference time.

This folder contains only the files that differ from upstream robomimic v0.5
(`ae5799f`), in `overlay/`:

| File | Change |
|---|---|
| `robomimic/algo/diffusion_policy.py` | `configure_sampler` and the TSR / CNS / MCMC / STEER / CREPE samplers |
| `robomimic/algo/crepe.py` | CREPE baseline (parallel tempering over diffusion time, no reward term) |
| `robomimic/scripts/run_trained_agent.py` | `--sampler`, `--lam_start`, `--lam_end`, `--n_replicas` and MCMC flags |
| `robomimic/scripts/subsample_train.py` | `train.py` with `--subsample_ratio`, which keeps a random fraction of training demos |
| `robomimic/exps/templates/diffusion_policy.json` | `seq_length=16`, `frame_stack=2`, `hdf5_load_next_obs=false` |

## Installation

```bash
conda create -n steer-robomimic python=3.9
conda activate steer-robomimic

bash setup.sh                       # clones robomimic at ae5799f into external/robomimic and applies the overlay
pip install -e external/robomimic
pip install robosuite==1.5.1
pip install -e ..                   # the steer package
```

Download the low-dimensional proficient-human datasets:

```bash
python external/robomimic/robomimic/scripts/download_datasets.py \
    --download_dir datasets --tasks lift can square tool_hang \
    --dataset_types ph --hdf5_types low_dim
```

## Training

`scripts/train.sh` trains a diffusion policy for 2000 epochs on a random fraction of the training
demos, keeping the full validation split:

```bash
bash scripts/train.sh lift      0.1  datasets/lift/ph/low_dim_v15.hdf5      checkpoints
bash scripts/train.sh can       0.1  datasets/can/ph/low_dim_v15.hdf5       checkpoints
bash scripts/train.sh square    0.75 datasets/square/ph/low_dim_v15.hdf5    checkpoints
bash scripts/train.sh tool_hang 0.75 datasets/tool_hang/ph/low_dim_v15.hdf5 checkpoints
```

The subset is written to the dataset as a filter key named `train_sub<ratio>_seed<seed>`. It is a
deterministic function of the ratio and the training seed. Concurrent jobs can safely share a dataset.

## Evaluation

`scripts/eval.sh` runs every sampler for 5 seeds × 50 rollouts with horizon 700:

```bash
bash scripts/eval.sh square checkpoints/square_diffusion_subsample0.75/<timestamp>/models/model_epoch_2000.pth
```

A single run looks like:

```bash
python external/robomimic/robomimic/scripts/run_trained_agent.py \
    --agent model_epoch_2000.pth --n_rollouts 50 --horizon 700 --seed 1 \
    --sampler steer --lam_start 0.9 --lam_end 1.1 --n_replicas 4
```

Settings used in the paper:

| Task | Demos kept | `lam_start` | `lam_end` | STEER `n_replicas` |
|---|---|---|---|---|
| Lift | 10% | 0.9 | 1.1 | 4 |
| Can | 10% | 0.9 | 1.1 | 4 |
| Square | 75% | 0.9 | 1.02 | 6 |
| Tool Hang | 75% | 0.9 | 1.1 | 4 |

`vanilla` uses `lam_start=1.0`. TSR, CNS, MCMC and CREPE use `lam_start=0.9` with a single replica.

## Samplers

| `--sampler` | Description |
|---|---|
| `vanilla` | Standard DDPM sampling |
| `tsr` | Temporal score rescaling at `lam_start` |
| `steer` | `n_replicas` rungs from `lam_start` to `lam_end` with replica exchange; returns the `lam_start` rung |
| `cns` | DDPM with the injected noise scaled by `1/sqrt(lam_start)` |
| `mcmc` | `--n_langevin_steps` Langevin updates on the score tempered by `lam_start` per timestep (`--step_size`, default `beta_t`) |
| `crepe` | CREPE parallel-tempering refinement of a vanilla sample |

`--seed` also seeds a dedicated generator for all sampling noise and swap decisions, so rollouts are
reproducible for a fixed seed.

## Acknowledgements

Built on [robomimic](https://github.com/ARISE-Initiative/robomimic) (MIT License) and
[robosuite](https://github.com/ARISE-Initiative/robosuite). Please cite their work alongside ours.
