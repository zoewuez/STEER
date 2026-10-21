"""
CREPE (Control with REPlica Exchange) refinement for Diffusion Policy.

A single ancestral sample is re-noised into a ladder over the scheduler's timesteps, and
checkerboard parallel tempering sweeps exchange states between neighbouring noise levels.
There is no reward or verifier term, so the acceptance test is the diffusion density ratio
alone. The lowest-noise levels can be excluded from swapping and finished with plain
ancestral denoising, since acceptance collapses as the posterior std approaches zero.
"""
import math
import torch


def _log_norm_prob(x, mu, std):
    std = std.expand(x.shape)
    dims = tuple(range(1, x.dim()))
    return -0.5 * ((x - mu) / std).pow(2).sum(dim=dims) - (torch.log(std) + 0.5 * math.log(2 * math.pi)).sum(dim=dims)


def _posterior_std(alpha_t, alpha_t_prev, eta):
    return eta * ((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)) ** 0.5


def predict_xt_forward(x_prev, alpha_t, alpha_t_prev, generator=None):
    """
    Sample x_t ~ q(x_t | x_prev) for the noisier level @alpha_t.
    """
    alpha_ratio = alpha_t / alpha_t_prev
    noise = torch.randn(x_prev.shape, device=x_prev.device, dtype=x_prev.dtype, generator=generator)
    return torch.sqrt(alpha_ratio) * x_prev + torch.sqrt(1 - alpha_ratio) * noise


def predict_x_prev_from_x0(xt, x0, alpha_prod_t, alpha_prod_t_prev, eta, t_is_zero, generator=None):
    """
    DDIM step (Song et al. 2021, Eq. 12) from a predicted x0. eta=1 recovers DDPM.
    """
    eps = (xt - alpha_prod_t ** 0.5 * x0) / (1 - alpha_prod_t) ** 0.5
    sigma = _posterior_std(alpha_prod_t, alpha_prod_t_prev, eta)
    pred_sample_direction = (1 - alpha_prod_t_prev - sigma ** 2) ** 0.5 * eps
    pred_x0_direction = (xt - (1 - alpha_prod_t) ** 0.5 * eps) / (alpha_prod_t ** 0.5)
    prev_sample = alpha_prod_t_prev ** 0.5 * pred_x0_direction + pred_sample_direction
    if eta > 0 and not t_is_zero:
        noise = torch.randn(xt.shape, device=xt.device, dtype=xt.dtype, generator=generator)
        prev_sample = prev_sample + sigma * noise
    return prev_sample


def _sweep(ladder, alphas, ts, eta, parity, model_fn, n_active_levels, generator=None):
    for i_hi in range(parity, n_active_levels - 1, 2):
        i_lo = i_hi + 1
        x_hi, x_lo = ladder[i_hi], ladder[i_lo]
        alpha_hi, alpha_lo = alphas[i_hi], alphas[i_lo]
        t_hi = ts[i_hi]

        eps_hi = model_fn(x_hi, t_hi)
        x0_hi = (x_hi - torch.sqrt(1 - alpha_hi) * eps_hi) / torch.sqrt(alpha_hi)

        x_fwd = predict_xt_forward(x_lo, alpha_hi, alpha_lo, generator=generator)
        x_bwd = predict_x_prev_from_x0(
            x_hi, x0_hi, alpha_hi, alpha_lo, eta,
            t_is_zero=(float(t_hi) == 0.0), generator=generator,
        )

        sigma_hi = _posterior_std(alpha_hi, alpha_lo, eta)
        log_fwd_density = _log_norm_prob(x_fwd, x_hi, sigma_hi)
        log_bwd_density = _log_norm_prob(x_bwd, x_lo, sigma_hi)

        log_alpha = torch.clamp(log_fwd_density - log_bwd_density, max=0.0)
        accept = torch.rand(log_alpha.shape, device=log_alpha.device, generator=generator) < torch.exp(log_alpha)
        accept_view = accept.view(-1, *([1] * (x_lo.dim() - 1)))

        ladder[i_hi] = torch.where(accept_view, x_fwd, x_hi)
        ladder[i_lo] = torch.where(accept_view, x_bwd, x_lo)

    return ladder


def refine_chain(model_fn, x0, alphas_cumprod, ts, eta=1.0, n_iterations=2, generator=None, stabilize_last_n=0):
    """
    Refine a fully denoised sample with CREPE and return the refined t=0 state.

    Args:
        model_fn (callable): maps (x, t) to the predicted noise
        x0 (torch.Tensor): ancestral sample at t=0
        alphas_cumprod (torch.Tensor): scheduler alphas_cumprod, indexed by timestep
        ts (torch.Tensor): scheduler timesteps, highest noise first
        eta (float): DDIM eta for backward proposals
        n_iterations (int): number of checkerboard sweeps, alternating parity
        generator (torch.Generator): optional generator for all randomness
        stabilize_last_n (int): number of lowest-noise levels excluded from swapping
    """
    n_levels = len(ts)
    alphas = [alphas_cumprod[int(t)] for t in ts]

    ladder = [None] * n_levels
    ladder[-1] = x0
    for i in range(n_levels - 2, -1, -1):
        ladder[i] = predict_xt_forward(ladder[i + 1], alphas[i], alphas[i + 1], generator=generator)

    n_active_levels = max(1, n_levels - stabilize_last_n)
    for it in range(n_iterations):
        ladder = _sweep(ladder, alphas, ts, eta, it % 2, model_fn, n_active_levels, generator=generator)

    x = ladder[n_active_levels - 1]
    for i in range(n_active_levels, n_levels):
        t_cur, alpha_cur, alpha_next = ts[i - 1], alphas[i - 1], alphas[i]
        eps = model_fn(x, t_cur)
        x0_est = (x - torch.sqrt(1 - alpha_cur) * eps) / torch.sqrt(alpha_cur)
        x = predict_x_prev_from_x0(x, x0_est, alpha_cur, alpha_next, eta, t_is_zero=(float(t_cur) == 0.0), generator=generator)

    return x
