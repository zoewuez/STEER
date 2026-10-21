"""
Code for sampling from diffusion models
"""
import json
import os
import multiprocessing as mp
from pathlib import Path
import tempfile
import logging
from typing import *

from tqdm.auto import tqdm

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import default_collate
from huggingface_hub import snapshot_download

from foldingdiff import datasets as dsets
from foldingdiff import beta_schedules, modelling, utils, sampling, tmalign
from foldingdiff import angles_and_coords as ac

from steer import init_temp_idx, lambda_ladder, rescale, select_rung, swap

SAMPLERS = ("vanilla", "tsr", "cns", "mcmc", "steer")
LADDER_SAMPLERS = ("tsr", "steer")

# (min length inclusive, max length exclusive, n_replicas)
REPLICA_SCHEDULE: List[Tuple[int, int, int]] = [
    (50, 60, 4),
    (60, 70, 5),
    (70, 80, 6),
    (80, 90, 7),
    (90, 100, 8),
    (100, 110, 9),
    (110, 129, 10),
]


def parse_replica_schedule(spec: Optional[str]) -> Optional[List[Tuple[int, int, int]]]:
    """
    Parse "none", "default", or explicit bins "lo:hi:n,lo:hi:n,..." (hi exclusive)
    """
    if spec is None or spec.strip().lower() in ("", "none"):
        return None
    if spec.strip().lower() == "default":
        return list(REPLICA_SCHEDULE)
    bins = []
    for chunk in spec.split(","):
        lo, hi, k = (int(v) for v in chunk.split(":"))
        if lo >= hi or k < 1:
            raise ValueError(f"Invalid replica schedule bin {chunk}")
        bins.append((lo, hi, k))
    bins.sort()
    for (_, hi_a, _), (lo_b, _, _) in zip(bins, bins[1:]):
        if lo_b < hi_a:
            raise ValueError(f"Overlapping replica schedule bins in {spec}")
    return bins


def n_replicas_for_length(
    length: int, schedule: Sequence[Tuple[int, int, int]] = REPLICA_SCHEDULE
) -> int:
    """
    Number of replicas for a given length; lengths outside the schedule are clamped
    """
    for lo, hi, k in schedule:
        if lo <= length < hi:
            return k
    return schedule[0][2] if length < schedule[0][0] else schedule[-1][2]


@torch.no_grad()
def mala_step(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    attn_mask: torch.Tensor,
    alpha_prod_t: torch.Tensor,
    step_size: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    One Metropolis-adjusted Langevin step at the current noise level, applied
    independently to each item in the batch
    """
    sqrt_one_minus_a = torch.sqrt(torch.clamp(1 - alpha_prod_t, min=1e-8))
    per_sample = lambda v: v.reshape(v.shape[0], -1).sum(dim=1)

    score_x = -model(x, t, attention_mask=attn_mask) / sqrt_one_minus_a
    z = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
    x_prop = x + (step_size / 2.0) * score_x + (step_size**0.5) * z
    score_prop = -model(x_prop, t, attention_mask=attn_mask) / sqrt_one_minus_a

    log_p_diff = 0.5 * per_sample((score_x + score_prop) * (x_prop - x))
    mu_x = x + (step_size / 2.0) * score_x
    mu_prop = x_prop + (step_size / 2.0) * score_prop
    fwd = per_sample((x_prop - mu_x) ** 2)
    bwd = per_sample((x - mu_prop) ** 2)
    log_q_diff = -(1.0 / (2.0 * step_size)) * (bwd - fwd)

    accept_prob = torch.exp(torch.clamp(log_p_diff + log_q_diff, max=0.0))
    u = torch.rand(
        accept_prob.shape, dtype=accept_prob.dtype, device=x.device, generator=generator
    )
    accept = (u < accept_prob).view(-1, *[1] * (x.dim() - 1))
    return torch.where(accept, x_prop, x)


@torch.no_grad()
def p_sample(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    seq_lens: Sequence[int],
    t_index: torch.Tensor,
    betas: torch.Tensor,
    sampler: str = "vanilla",
    lam_ladder: Optional[torch.Tensor] = None,
    temp_idx: Optional[torch.Tensor] = None,
    mala_step_size: float = 2e-5,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Sample the given timestep. Note that this _may_ fall off the manifold if we just
    feed the output back into itself repeatedly, so we need to perform modulo on it
    (see p_sample_loop)

    For "tsr" and "steer", lam_ladder and temp_idx describe the replica ladder held
    in the batch; "steer" updates temp_idx in place.
    """
    # Calculate alphas and betas
    alpha_beta_values = beta_schedules.compute_alphas(betas)
    sqrt_recip_alphas = 1.0 / torch.sqrt(alpha_beta_values["alphas"])

    # Select based on time
    t_unique = torch.unique(t)
    assert len(t_unique) == 1, f"Got multiple values for t: {t_unique}"
    t_index = t_unique.item()
    sqrt_recip_alphas_t = sqrt_recip_alphas[t_index]
    betas_t = betas[t_index]
    alphas_cumprod_t = alpha_beta_values["alphas_cumprod"][t_index]
    sqrt_one_minus_alphas_cumprod_t = alpha_beta_values[
        "sqrt_one_minus_alphas_cumprod"
    ][t_index]

    # Create the attention mask
    attn_mask = torch.zeros(x.shape[:2], device=x.device)
    for i, length in enumerate(seq_lens):
        attn_mask[i, :length] = 1.0

    if sampler == "mcmc":
        x = mala_step(
            model, x, t, attn_mask, alphas_cumprod_t, mala_step_size, generator
        )

    epsilon = model(x, t, attention_mask=attn_mask)
    if sampler == "steer":
        score = -epsilon / sqrt_one_minus_alphas_cumprod_t
        swap(x, score, alphas_cumprod_t, lam_ladder, temp_idx, t_index, generator)
    if sampler in LADDER_SAMPLERS:
        epsilon = rescale(epsilon, alphas_cumprod_t, lam_ladder, temp_idx)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * epsilon / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = alpha_beta_values["posterior_variance"][t_index]
        noise = torch.randn_like(x)
        if sampler == "cns":
            noise *= lam_ladder[0].sqrt()
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def p_sample_loop(
    model: nn.Module,
    lengths: Sequence[int],
    noise: torch.Tensor,
    timesteps: int,
    betas: torch.Tensor,
    is_angle: Union[bool, List[bool]] = [False, True, True, True],
    disable_pbar: bool = False,
    sampler: str = "vanilla",
    lam_start: float = 1.0,
    lam_end: float = 1.0,
    mala_step_size: float = 2e-5,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Returns a tensor of shape (timesteps, batch_size, seq_len, n_ft)

    For "tsr" and "steer" the batch is a single replica ladder, and only the
    trajectory at lam_start is returned (batch_size 1)
    """
    if sampler not in SAMPLERS:
        raise ValueError(f"Unknown sampler {sampler}, expected one of {SAMPLERS}")
    device = next(model.parameters()).device
    b = noise.shape[0]
    img = noise.to(device)
    n_replicas = b if sampler in LADDER_SAMPLERS else 1
    lam_ladder = lambda_ladder(lam_start, lam_end, n_replicas, device=device)
    temp_idx = init_temp_idx(n_replicas, 1, device=device)
    # Report metrics on starting noise
    # amin and amax support reducing on multiple dimensions
    logging.info(
        f"Starting from noise {noise.shape} with angularity {is_angle} and range {torch.amin(img, dim=(0, 1))} - {torch.amax(img, dim=(0, 1))} using {device}"
    )

    imgs = []

    for i in tqdm(
        reversed(range(0, timesteps)),
        desc="sampling loop time step",
        total=timesteps,
        disable=disable_pbar,
    ):
        # Shape is (batch, seq_len, 4)
        img = p_sample(
            model=model,
            x=img,
            t=torch.full((b,), i, device=device, dtype=torch.long),  # time vector
            seq_lens=lengths,
            t_index=i,
            betas=betas,
            sampler=sampler,
            lam_ladder=lam_ladder,
            temp_idx=temp_idx,
            mala_step_size=mala_step_size,
            generator=generator,
        )

        # Wrap if angular
        if isinstance(is_angle, bool):
            if is_angle:
                img = utils.modulo_with_wrapped_range(
                    img, range_min=-torch.pi, range_max=torch.pi
                )
        else:
            assert len(is_angle) == img.shape[-1]
            for j in range(img.shape[-1]):
                if is_angle[j]:
                    img[:, :, j] = utils.modulo_with_wrapped_range(
                        img[:, :, j], range_min=-torch.pi, range_max=torch.pi
                    )
        imgs.append(img.cpu())
    if sampler in LADDER_SAMPLERS:
        imgs = [select_rung(img, temp_idx.cpu(), rung=0) for img in imgs]
    return torch.stack(imgs)


def sample(
    model: nn.Module,
    train_dset: dsets.NoisedAnglesDataset,
    n: int = 10,
    sweep_lengths: Optional[Tuple[int, int]] = (50, 128),
    batch_size: int = 512,
    feature_key: str = "angles",
    disable_pbar: bool = False,
    trim_to_length: bool = True,  # Trim padding regions to reduce memory
    sampler: str = "vanilla",
    n_replicas: int = 1,
    replica_schedule: Optional[Sequence[Tuple[int, int, int]]] = None,
    lam_start: float = 1.0,
    lam_end: float = 1.0,
    mala_step_size: float = 2e-5,
    generator: Optional[torch.Generator] = None,
) -> List[np.ndarray]:
    """
    Sample from the given model. Use the train_dset to generate noise to sample
    sequence lengths. Returns a list of arrays, shape (timesteps, seq_len, fts).
    If sweep_lengths is set, we generate n items per length in the sweep range

    train_dset object must support:
    - sample_noise - provided by NoisedAnglesDataset
    - timesteps - provided by NoisedAnglesDataset
    - alpha_beta_terms - provided by NoisedAnglesDataset
    - feature_is_angular - provided by *wrapped dataset* under NoisedAnglesDataset
    - pad - provided by *wrapped dataset* under NoisedAnglesDataset
    And optionally, sample_length()

    Samplers:
    - vanilla: ancestral DDPM sampling
    - tsr: temporal score rescaling at lam_start
    - steer: a ladder of n_replicas temperatures from lam_start to lam_end with
      replica exchange; the sample at lam_start is kept
    - cns: posterior noise scaled by sqrt(lam_start)
    - mcmc: one MALA corrector step before each denoising step
    For tsr and steer each example is sampled as its own ladder, so batch_size is
    ignored. If replica_schedule is given, the ladder size depends on length.
    """
    # Process each batch
    if sweep_lengths is not None:
        sweep_min, sweep_max = sweep_lengths
        if not sweep_min < sweep_max:
            raise ValueError(
                f"Minimum length {sweep_min} must be less than maximum {sweep_max}"
            )
        logging.info(
            f"Sweeping from {sweep_min}-{sweep_max} with {n} examples at each length"
        )
        lengths = []
        for l in range(sweep_min, sweep_max):
            lengths.extend([l] * n)
    else:
        lengths = [train_dset.sample_length() for _ in range(n)]
    if sampler in LADDER_SAMPLERS:
        ladder_size = lambda l: (
            n_replicas_for_length(l, replica_schedule) if replica_schedule else n_replicas
        )
        lengths_chunkified = [[l] * ladder_size(l) for l in lengths]
        logging.info(f"Sampling {len(lengths)} items as one replica ladder each")
    else:
        lengths_chunkified = [
            lengths[i : i + batch_size] for i in range(0, len(lengths), batch_size)
        ]
        logging.info(f"Sampling {len(lengths)} items in batches of size {batch_size}")
    retval = []
    for this_lengths in lengths_chunkified:
        batch = len(this_lengths)
        # Sample noise and sample the lengths
        noise = train_dset.sample_noise(
            torch.zeros((batch, train_dset.pad, model.n_inputs), dtype=torch.float32)
        )

        # Trim things that are beyond the length of what we are generating
        if trim_to_length:
            noise = noise[:, : max(this_lengths), :]

        # Produces (timesteps, batch_size, seq_len, n_ft)
        sampled = p_sample_loop(
            model=model,
            lengths=this_lengths,
            noise=noise,
            timesteps=train_dset.timesteps,
            betas=train_dset.alpha_beta_terms["betas"],
            is_angle=train_dset.feature_is_angular[feature_key],
            disable_pbar=disable_pbar,
            sampler=sampler,
            lam_start=lam_start,
            lam_end=lam_end,
            mala_step_size=mala_step_size,
            generator=generator,
        )
        # Gets to size (timesteps, seq_len, n_ft)
        trimmed_sampled = [
            sampled[:, i, :l, :].numpy()
            for i, l in enumerate(this_lengths[: sampled.shape[1]])
        ]
        retval.extend(trimmed_sampled)
    # Note that we don't use means variable here directly because we may need a subset
    # of it based on which features are active in the dataset. The function
    # get_masked_means handles this gracefully
    if (
        hasattr(train_dset, "dset")
        and hasattr(train_dset.dset, "get_masked_means")
        and train_dset.dset.get_masked_means() is not None
    ):
        logging.info(
            f"Shifting predicted values by original offset: {train_dset.dset.get_masked_means()}"
        )
        retval = [s + train_dset.dset.get_masked_means() for s in retval]
        # Because shifting may have caused us to go across the circle boundary, re-wrap
        angular_idx = np.where(train_dset.feature_is_angular[feature_key])[0]
        for s in retval:
            s[..., angular_idx] = utils.modulo_with_wrapped_range(
                s[..., angular_idx], range_min=-np.pi, range_max=np.pi
            )

    return retval


def sample_simple(
    model_dir: str, n: int = 10, sweep_lengths: Tuple[int, int] = (50, 128)
) -> List[pd.DataFrame]:
    """
    Simple wrapper on sample to automatically load in the model and dummy dataset
    Primarily for gradio integration
    """
    if utils.is_huggingface_hub_id(model_dir):
        model_dir = snapshot_download(model_dir)
    assert os.path.isdir(model_dir)

    with open(os.path.join(model_dir, "training_args.json")) as source:
        training_args = json.load(source)

    model = modelling.BertForDiffusionBase.from_dir(model_dir)
    if torch.cuda.is_available():
        model = model.to("cuda:0")

    dummy_dset = dsets.AnglesEmptyDataset.from_dir(model_dir)
    dummy_noised_dset = dsets.NoisedAnglesDataset(
        dset=dummy_dset,
        dset_key="coords" if training_args == "cart-cords" else "angles",
        timesteps=training_args["timesteps"],
        exhaustive_t=False,
        beta_schedule=training_args["variance_schedule"],
        nonangular_variance=1.0,
        angular_variance=training_args["variance_scale"],
    )

    sampled = sample(
        model, dummy_noised_dset, n=n, sweep_lengths=sweep_lengths, disable_pbar=True
    )
    final_sampled = [s[-1] for s in sampled]
    sampled_dfs = [
        pd.DataFrame(s, columns=dummy_noised_dset.feature_names["angles"])
        for s in final_sampled
    ]
    return sampled_dfs


def _score_angles(
    reconst_angles: pd.DataFrame, truth_angles: pd.DataFrame, truth_coords_pdb: str
) -> Tuple[float, float]:
    """
    Helper function to scores sets of angles
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        truth_path = Path(tmpdir) / "truth.pdb"
        reconst_path = Path(tmpdir) / "reconst.pdb"

        truth_pdb = ac.create_new_chain_nerf(str(truth_path), truth_angles)
        reconst_pdb = ac.create_new_chain_nerf(str(reconst_path), reconst_angles)

        # Calculate WRT the truth angles
        score = tmalign.run_tmalign(reconst_pdb, truth_pdb)

        score_coord = tmalign.run_tmalign(reconst_pdb, truth_coords_pdb)
    return score, score_coord


@torch.no_grad()
def get_reconstruction_error(
    model: nn.Module, dset, noise_timesteps: int = 250, bs: int = 512
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the reconstruction error when adding <noise_timesteps> noise to the idx-th
    item in the dataset.
    """
    device = next(model.parameters()).device
    model.eval()

    recont_angle_sets = []
    truth_angle_sets = []
    truth_pdb_files = []
    for idx_batch in tqdm(utils.seq_to_groups(list(range(len(dset))), bs)):
        batch = default_collate(
            [
                {
                    k: v.to(device)
                    for k, v in dset.__getitem__(idx, use_t_val=noise_timesteps).items()
                }
                for idx in idx_batch
            ]
        )
        img = batch["corrupted"].clone()
        assert img.ndim == 3

        # Record the actual files containing raw coordinates
        for i in idx_batch:
            truth_pdb_files.append(dset.filenames[i])

        # Run the diffusion model for noise_timesteps steps
        for i in tqdm(list(reversed(list(range(0, noise_timesteps))))):
            img = sampling.p_sample(
                model=model,
                x=img,
                t=torch.full((len(idx_batch),), fill_value=i, dtype=torch.long).to(
                    device
                ),
                seq_lens=batch["lengths"],
                t_index=i,
                betas=dset.alpha_beta_terms["betas"],
            )
            img = utils.modulo_with_wrapped_range(img)

        # Finished reconstruction, subset to lengths and add to running list
        for i, l in enumerate(batch["lengths"].squeeze()):
            recont_angle_sets.append(
                pd.DataFrame(img[i, :l].cpu().numpy(), columns=ac.EXHAUSTIVE_ANGLES)
            )
            truth_angle_sets.append(
                pd.DataFrame(
                    batch["angles"][i, :l].cpu().numpy(), columns=ac.EXHAUSTIVE_ANGLES
                )
            )

    # Get the reconstruction error as a TM score
    logging.info(
        f"Calculating TM scores for reconstruction error with {mp.cpu_count()} processes"
    )
    pool = mp.Pool(processes=mp.cpu_count())
    results = pool.starmap(
        _score_angles,
        zip(recont_angle_sets, truth_angle_sets, truth_pdb_files),
        chunksize=10,
    )
    pool.close()
    pool.join()
    scores, coord_scores = zip(*results)
    return np.array(scores), np.array(coord_scores)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    s = sample_simple("wukevin/foldingdiff_cath", n=1, sweep_lengths=(50, 51))
    for i, x in enumerate(s):
        print(x.shape)
        print(x)
