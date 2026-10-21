import torch

from steer import (
    init_temp_idx,
    lambda_ladder,
    rescale,
    score_scale,
    select_rung,
    sort_by_rung,
    swap,
)


def test_ladder_endpoints():
    for spacing in ("linear", "geometric", "geometric_gamma"):
        ladder = lambda_ladder(0.5, 2.0, 5, spacing=spacing)
        assert torch.isclose(ladder[0], torch.tensor(0.5))
        assert torch.isclose(ladder[-1], torch.tensor(2.0))
        assert torch.all(ladder[1:] > ladder[:-1])
    assert torch.allclose(lambda_ladder(0.7, 2.0, 1), torch.tensor([0.7]))


def test_score_scale_limits():
    assert score_scale(0.0, 3.0) == 1.0
    assert score_scale(1.0, 4.0) == 0.25


def test_swap_keeps_permutation():
    torch.manual_seed(0)
    n_replicas, n_chains = 6, 32
    ladder = lambda_ladder(0.3, 3.0, n_replicas)
    temp_idx = init_temp_idx(n_replicas, n_chains)
    for step in range(50):
        x = torch.randn(n_replicas * n_chains, 2)
        score = torch.randn(n_replicas * n_chains, 2)
        swap(x, score, torch.tensor(0.5), ladder, temp_idx, step)
        assert torch.equal(temp_idx.sort(dim=0).values, init_temp_idx(n_replicas, n_chains))
    assert not torch.equal(temp_idx, init_temp_idx(n_replicas, n_chains))


def test_equal_temperatures_always_swap():
    n_replicas = 4
    ladder = torch.ones(n_replicas)
    temp_idx = init_temp_idx(n_replicas)
    x = torch.randn(n_replicas, 3)
    swap(x, torch.randn_like(x), torch.tensor(0.5), ladder, temp_idx, step=0)
    assert temp_idx[:, 0].tolist() == [1, 0, 3, 2]


def test_rescale_and_readout():
    n_replicas, n_chains, k = 3, 2, 4
    ladder = lambda_ladder(0.5, 1.5, n_replicas)
    temp_idx = torch.tensor([[2, 0], [0, 1], [1, 2]])
    x = torch.arange(n_replicas * n_chains * k, dtype=torch.float32).view(-1, 1)

    scaled = rescale(torch.ones_like(x), 0.4, ladder, temp_idx).view(n_replicas, n_chains, k)
    for r in range(n_replicas):
        for c in range(n_chains):
            expected = score_scale(0.4, ladder[temp_idx[r, c]])
            assert torch.allclose(scaled[r, c], expected.expand(k))

    view = x.view(n_replicas, n_chains, k)
    cold = select_rung(x, temp_idx, 0).view(n_chains, k)
    assert torch.equal(cold[0], view[1, 0])
    assert torch.equal(cold[1], view[0, 1])

    ordered = sort_by_rung(x, temp_idx).view(n_replicas, n_chains, k)
    for rung in range(n_replicas):
        assert torch.equal(ordered[rung], select_rung(x, temp_idx, rung).view(n_chains, k))
