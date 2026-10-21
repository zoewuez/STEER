# STEER for protein backbone generation (FoldingDiff)

STEER applied to [FoldingDiff](https://github.com/microsoft/foldingdiff), a DDPM over backbone
torsion and bond angles. Each backbone is generated with a ladder of `n_replicas` temperatures
running from `lam_start` to `lam_end`. Every replica denoises with the temperature-rescaled noise
prediction for its rung. Neighbouring rungs propose replica-exchange swaps at every step, and the
replica at `lam_start` is kept.

## Layout

```
overlay/                     files added to or replacing upstream FoldingDiff
  foldingdiff/sampling.py    samplers: vanilla, tsr, steer, cns, mcmc
  foldingdiff/tmalign.py     parses both old (Chain_) and new (Structure_) TMalign output
  bin/sample.py              --sampler, --lam_start, --lam_end, --n_replicas, --replica_schedule
  bin/merge_sctm.py          collects scTM scores from a sweep into one CSV
setup.sh                     clones upstream at the pinned commit and applies the overlay
run_sweep.sh                 sampling + designability pipeline over length chunks
```

## Installation

```bash
bash setup.sh                       # clones to ./external/foldingdiff (or pass a path)
cd external/foldingdiff
conda env create -f environment.yml && conda activate foldingdiff
pip install -e .
pip install -e ../../..             # the steer package at the repository root
```

The upstream commit is `18c865338db937fbd2ffba1e4fdfbcfdb99de8b7`.

## Pretrained model

The scripts use the released CATH model [`wukevin/foldingdiff_cath`](https://huggingface.co/wukevin/foldingdiff_cath).
It is downloaded from the Hugging Face Hub on first use (the default for `bin/sample.py -m`).

## Sampling

Run these from `external/foldingdiff/bin`. Each command writes `sampled_angles/`, `sampled_pdb/`
and `sampler_config.json` to `--outdir`.

```bash
# STEER: 10 backbones per length 50-127, ladder 0.5 -> 0.6 with 10 replicas
python sample.py --outdir steer_out -n 10 -l 50 128 \
    --sampler steer --lam_start 0.5 --lam_end 0.6 --n_replicas 10

# STEER with a length-dependent ladder size (4 replicas at length 50 up to 10 at length 110+)
python sample.py --outdir steer_sched -n 10 -l 50 128 \
    --sampler steer --lam_start 0.5 --lam_end 0.6 --replica_schedule default

# Baselines
python sample.py --outdir vanilla_out -n 10 -l 50 128
python sample.py --outdir tsr_out -n 10 -l 50 128 --sampler tsr --lam_start 0.5
python sample.py --outdir cns_out -n 10 -l 50 128 --sampler cns --lam_start 0.5
python sample.py --outdir mcmc_out -n 10 -l 50 128 --sampler mcmc --mala_step_size 2e-5
```

| Flag | Default | Description |
| --- | --- | --- |
| `--sampler` | `vanilla` | `vanilla`, `tsr`, `steer`, `cns` or `mcmc` |
| `--lam_start` | `1.0` | target temperature (`tsr`, `steer`, `cns`) |
| `--lam_end` | `1.0` | hottest rung of the ladder (`steer`) |
| `--n_replicas` | `1` | replicas per ladder (`tsr`, `steer`) |
| `--replica_schedule` | `none` | `none`, `default`, or bins `lo:hi:n,...` with `hi` exclusive; overrides `--n_replicas` |
| `--mala_step_size` | `2e-5` | Langevin step size (`mcmc`) |
| `--seed` | upstream default | seeds both the sampler and the swap decisions |

With `tsr` and `steer`, each backbone is sampled as its own ladder, so `--batchsize` is ignored.
Sampling cost grows linearly with the ladder size.

Samplers:

- `tsr`: temporal score rescaling at `lam_start`, with no exchange.
- `steer`: rescaling on every rung, plus replica exchange between neighbouring rungs.
- `cns`: scales the posterior noise by `sqrt(lam_start)`.
- `mcmc`: runs one Metropolis-adjusted Langevin step before each denoising step.

## Designability (scTM)

Designability follows the FoldingDiff protocol:

1. Generate 8 sequences per backbone with ProteinMPNN.
2. Fold them with OmegaFold.
3. Take the best TM-score to the generated backbone as the scTM score. Backbones with scTM ≥ 0.5
   count as designable.

ProteinMPNN and OmegaFold each need their own environment; follow the upstream FoldingDiff
README to set them up. `TMalign` must be on `PATH`.

`run_sweep.sh` runs the full pipeline for one sampler and one `lam_start`. Lengths are split into
chunks whose ladder grows with length, from 4 replicas for lengths 50-59 up to 10 for 110-127,
with `lam_end = lam_start + LAM_STEP * n_replicas`.

```bash
bash run_sweep.sh results/steer_lam0.5 steer 0.5
LAM_STEP=0.01 NUM_PER_LENGTH=10 bash run_sweep.sh results/tsr_lam0.5 tsr 0.5
```

The reported experiments sweep `lam_start` over 0.5, 0.6, 0.7, 0.8 and 0.9 with `LAM_STEP=0.01`.

Environment names are set through `FOLDINGDIFF_ENV`, `PROTEINMPNN_ENV` and `OMEGAFOLD_ENV`. Set
`PROTFID_ENV` to also compute a pooled FID with `protfid`. The merged per-backbone scores are
written to `combined_sctm.csv`. To merge several sweeps:

```bash
python external/foldingdiff/bin/merge_sctm.py results/* -o all_sctm.csv
```

## Acknowledgements

This builds on FoldingDiff (Wu et al., *Protein structure generation via folding diffusion*,
Nature Communications 2024), released under the MIT license.
