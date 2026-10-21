"""Tempered score rescaling with replica exchange.

A ladder of ``n_replicas`` temperatures ``lam_0, ..., lam_{R-1}`` is simulated in
parallel. Each replica denoises with the noise prediction rescaled for its rung
(temporal score rescaling), and at every step neighbouring rungs propose to
exchange temperatures through a Metropolis test.

Tensors that carry replicas have a leading dimension of ``n_replicas * n_chains * k``
laid out replica-major, i.e. they can be viewed as ``(n_replicas, n_chains, k, ...)``.
Independent ladders are indexed by ``n_chains``; ``k`` groups samples that share a
single swap decision. ``temp_idx[r, c]`` is the rung currently held by slot ``r`` of
chain ``c``. Swaps relabel slots and never move data.
"""

import math

import torch


def score_scale(alpha, lam):
    return 1.0 / (alpha * lam + (1.0 - alpha))


def lambda_ladder(lam_start, lam_end, n_replicas, spacing="linear", ref_snr=1.0,
                  device=None, dtype=torch.float32):
    if n_replicas == 1:
        return torch.tensor([lam_start], device=device, dtype=dtype)
    if spacing == "linear":
        return torch.linspace(lam_start, lam_end, n_replicas, device=device, dtype=dtype)
    if spacing == "geometric":
        return torch.logspace(math.log10(lam_start), math.log10(lam_end), n_replicas,
                              device=device, dtype=dtype)
    if spacing == "geometric_gamma":
        a = ref_snr / (1.0 + ref_snr)
        g = torch.logspace(math.log10(a * lam_start + 1 - a), math.log10(a * lam_end + 1 - a),
                           n_replicas, device=device, dtype=dtype)
        return (g - (1 - a)) / a
    raise ValueError(f"unknown spacing {spacing!r}")


def init_temp_idx(n_replicas, n_chains=1, device=None):
    return torch.arange(n_replicas, device=device).unsqueeze(1).repeat(1, n_chains)


def slot_of_rung(temp_idx, rung):
    return (temp_idx == rung).float().argmax(dim=0)


def _expand_to(values, x):
    n_replicas, n_chains = values.shape
    k = x.shape[0] // (n_replicas * n_chains)
    values = values.unsqueeze(-1).expand(n_replicas, n_chains, k).reshape(-1)
    return values.view(-1, *([1] * (x.dim() - 1)))


def rescale(eps, alpha, lam_ladder, temp_idx):
    factor = score_scale(alpha, lam_ladder[temp_idx])
    return eps * _expand_to(factor, eps).to(eps.dtype)


@torch.no_grad()
def swap(x, score, alpha, lam_ladder, temp_idx, step, generator=None):
    n_replicas, n_chains = temp_idx.shape
    x = x.reshape(n_replicas, n_chains, -1).float()
    score = score.reshape(n_replicas, n_chains, -1).float()
    chains = torch.arange(n_chains, device=temp_idx.device)
    rand_device = generator.device if generator is not None else x.device

    for rung in range(step % 2, n_replicas - 1, 2):
        lo, hi = slot_of_rung(temp_idx, rung), slot_of_rung(temp_idx, rung + 1)
        dx = x[lo, chains] - x[hi, chains]
        ds = score[lo, chains] + score[hi, chains]
        dc = score_scale(alpha, lam_ladder[rung]) - score_scale(alpha, lam_ladder[rung + 1])
        log_accept = (-0.5 * dc * (ds * dx).sum(dim=-1)).clamp(max=0.0)

        u = torch.rand(n_chains, generator=generator, device=rand_device).to(x.device)
        accept = u < log_accept.exp()
        temp_idx[lo[accept], chains[accept]] = rung + 1
        temp_idx[hi[accept], chains[accept]] = rung

    return temp_idx


def _replica_view(x, temp_idx):
    n_replicas, n_chains = temp_idx.shape
    return x.reshape(n_replicas, n_chains, -1, *x.shape[1:])


def select_rung(x, temp_idx, rung=0):
    view = _replica_view(x, temp_idx)
    chains = torch.arange(view.shape[1], device=x.device)
    return view[slot_of_rung(temp_idx, rung), chains].reshape(-1, *x.shape[1:])


def sort_by_rung(x, temp_idx):
    view = _replica_view(x, temp_idx)
    chains = torch.arange(view.shape[1], device=x.device)
    return view[temp_idx.argsort(dim=0), chains].reshape(-1, *x.shape[1:])
