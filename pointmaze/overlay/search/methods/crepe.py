import math

import torch
from diffuser.models.helpers import apply_conditioning
from search.methods.base_guidance import BaseGuidance
from search.search_policy import SearchPolicy


def log_normal(x, mu, std):
    std = std.expand(x.shape)
    dims = tuple(range(1, x.dim()))
    return -0.5 * ((x - mu) / std).pow(2).sum(dim=dims) - (torch.log(std) + 0.5 * math.log(2 * math.pi)).sum(dim=dims)


class CREPEPolicy(SearchPolicy):
    """Ancestral sampling followed by reward-tilted parallel tempering over diffusion time (CREPE)."""

    def posterior_std(self, i):
        alpha_t = self.alpha_prod_ts[i]
        alpha_t_prev = self.alpha_prod_t_prevs[i]
        return self.eta * ((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)) ** 0.5

    def crepe_iteration(self, x, cond, guidance: BaseGuidance, parity):
        n_levels = len(self.ts)
        horizon = self.diffusion.horizon
        action_dim = self.diffusion.action_dim
        mask_last = max(0, self.args.crepe_mask_last)

        rho = self.args.crepe_reward_rho
        v_min = self.args.crepe_reward_min ** (1.0 / rho)
        v_max = self.args.crepe_reward_max ** (1.0 / rho)
        progress = torch.arange(n_levels, device=x.device, dtype=torch.float32) / (n_levels - 1)
        reward_weight = (v_min + progress * (v_max - v_min)) ** rho

        def predict_x0(xi, i):
            x0 = self.diffusion.model(xi, cond, self.ts[i].repeat(xi.shape[0]))
            return apply_conditioning(x0, cond, action_dim)

        def log_reward_from_x0(x0, i):
            if mask_last > 0:
                x0 = x0[:, : horizon - mask_last]
            logr = guidance.guider.get_guidance(
                x0, return_logp=True, check_grad=False, cond=cond, post_process=self.unnormalize
            )
            return reward_weight[i] * logr

        def log_reward(xi, i):
            return log_reward_from_x0(predict_x0(xi, i), i)

        levels = [None] * n_levels
        levels[-1] = x
        for i in range(n_levels - 2, -1, -1):
            x_up = guidance._predict_xt(levels[i + 1], self.alpha_prod_ts[i], self.alpha_prod_t_prevs[i], cond=cond)
            levels[i] = apply_conditioning(x_up, cond, action_dim)
        ladder = torch.stack(levels, dim=0)

        nfe = 0
        with torch.no_grad():
            for i_hi in range(parity, n_levels - 1, 2):
                i_lo = i_hi + 1
                x_hi, x_lo = ladder[i_hi], ladder[i_lo]
                alpha_t, alpha_t_prev = self.alpha_prod_ts[i_hi], self.alpha_prod_t_prevs[i_hi]

                logr_lo = log_reward(x_lo, i_lo)
                x0_hi = predict_x0(x_hi, i_hi)
                logr_hi = log_reward_from_x0(x0_hi, i_hi)

                x_fwd = apply_conditioning(guidance._predict_xt(x_lo, alpha_t, alpha_t_prev, cond=cond), cond, action_dim)
                x_bwd = guidance._predict_x_prev_from_zero(
                    x_hi, x0_hi, alpha_t, alpha_t_prev, self.eta, self.ts[i_hi],
                    cond=cond, post_process=self.unnormalize,
                )
                x_bwd = apply_conditioning(x_bwd, cond, action_dim)

                logr_fwd = log_reward(x_fwd, i_hi)
                logr_bwd = log_reward(x_bwd, i_lo)
                nfe += 4

                std = self.posterior_std(i_hi)
                log_alpha = (logr_bwd - logr_lo) + (logr_fwd - logr_hi) \
                    + (log_normal(x_fwd, x_hi, std) - log_normal(x_bwd, x_lo, std))
                accept = torch.rand_like(log_alpha) < torch.exp(torch.clamp(log_alpha, max=0.0))
                accept = accept.view(-1, *([1] * (x_lo.dim() - 1)))

                ladder[i_hi] = torch.where(accept, x_fwd, x_hi)
                ladder[i_lo] = torch.where(accept, x_bwd, x_lo)

        return ladder[-1], nfe

    def sample(self, cond, guidance: BaseGuidance, **kwargs):
        B = self.per_sample_batch_size
        x, extras = super().sample(cond, guidance, **kwargs)
        total_compute = extras['compute']

        cond = {k: (v[:1] if hasattr(v, "__getitem__") else v) for k, v in cond.items()}
        x = x.repeat(B, 1, 1)
        cond = {
            k: (v.repeat(B, *([1] * (v.dim() - 1))) if hasattr(v, "__getitem__") else v)
            for k, v in cond.items()
        }

        for it in range(self.args.crepe_iterations):
            x, nfe = self.crepe_iteration(x, cond, guidance, parity=it % 2)
            total_compute += nfe

        log_probs = guidance.guider.get_guidance(
            x, return_logp=True, check_grad=False, cond=cond, post_process=self.unnormalize
        )
        x = x[log_probs.argmax(dim=0)].unsqueeze(0)

        return x, {'compute': total_compute}
