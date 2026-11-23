"""E7 CPU Phase 1 tests: rank expansion (A1-A7) and thin/core-SVD reduction (B1-B5).

Authoritative spec: results/e7/design_20261120/v5_final_reconciliation/
EXPANSION_AND_SVD_TEST_SPEC.md and OPTIMIZER_LIFECYCLE_FINAL.md.

Conventions verified against PEFT 0.21.2: A is [rank, d_in], B is
[d_out, rank], non-rslora scaling is alpha/r with the E7 protocol
alpha = 2*rank (effective scaling 2, rank-invariant).

All tests are CPU-only, deterministic (seeded generators / manual_seed),
and use small synthetic tensors. No model downloads, no datasets, no
MPS/CUDA. Tolerances: float64 for the strict 1e-12 / 1e-6 criteria;
the float32 path is tested separately with a documented 1e-6 absolute
tolerance (float32 cannot guarantee 1e-12 in general, although the
expansion boundary is exact in both dtypes because appended B columns
are exactly zero).
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cycleft.e7.rank_transition import (
    alpha_for_rank,
    effective_scaling,
    expand_lora_factors,
    reduce_lora_factors,
)

SEED = 42
F32_TOL = 1e-6  # documented dtype-appropriate tolerance for the float32 path
F64_STRICT = 1e-12  # float64 strict tolerance (B4 QR-vs-direct comparison)
F64_TAIL = 1e-6  # float64 tail-energy agreement (B1/B4)


def gen(seed: int = SEED) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def make_factors(r_old: int, d_in: int, d_out: int, dtype=torch.float32, seed: int = SEED):
    """Random factors with well-conditioned, non-degenerate spectrum."""
    g = gen(seed)
    A = torch.randn(r_old, d_in, generator=g, dtype=dtype)
    B = torch.randn(d_out, r_old, generator=g, dtype=dtype)
    return A, B


def effective_update(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """E = (B @ A) * (alpha/rank) with alpha = 2*rank -> scaling exactly 2."""
    r = A.shape[0]
    return (B @ A) * effective_scaling(r)


# ---------------------------------------------------------------------------
# A-tests: zero-preserving expansion
# ---------------------------------------------------------------------------


def test_a1_component_preservation():
    torch.manual_seed(SEED)
    A, B = make_factors(r_old=4, d_in=16, d_out=12)
    A_new, B_new = expand_lora_factors(A, B, r_new=8, generator=gen())
    assert torch.equal(A_new[:4], A), "old A rows not bitwise identical"
    assert torch.equal(B_new[:, :4], B), "old B columns not bitwise identical"
    # new A rows finite and inside the Kaiming(a=sqrt(5)) bound 1/sqrt(fan_in=d_in)
    assert torch.isfinite(A_new[4:]).all()
    bound = 1.0 / (16**0.5)
    assert A_new[4:].abs().max() <= bound
    # new B columns exactly zero
    assert torch.all(B_new[:, 4:] == 0)


def test_a2_output_continuity():
    torch.manual_seed(SEED)
    A64, B64 = make_factors(r_old=4, d_in=16, d_out=12, dtype=torch.float64)
    x64 = torch.randn(5, 16, generator=gen(), dtype=torch.float64)
    pre64 = effective_update(A64, B64) @ x64.T
    A64n, B64n = expand_lora_factors(A64, B64, r_new=8, generator=gen())
    post64 = effective_update(A64n, B64n) @ x64.T
    assert torch.allclose(pre64, post64, atol=1e-12, rtol=0.0)

    A32, B32 = make_factors(r_old=4, d_in=16, d_out=12, dtype=torch.float32)
    x32 = torch.randn(5, 16, generator=gen(), dtype=torch.float32)
    pre32 = effective_update(A32, B32) @ x32.T
    A32n, B32n = expand_lora_factors(A32, B32, r_new=8, generator=gen())
    post32 = effective_update(A32n, B32n) @ x32.T
    assert torch.allclose(pre32, post32, atol=F32_TOL, rtol=0.0)
    # the boundary is in fact exact in float32 as well (appended columns are
    # exactly zero); documented tolerance exists because float32 in general
    # cannot guarantee 1e-12, not because this boundary needs it.
    assert torch.equal(pre32, post32)


def test_a3_new_b_gradient_nonzero():
    torch.manual_seed(SEED)
    A, B = make_factors(r_old=2, d_in=8, d_out=6)
    A_new, B_new = expand_lora_factors(A, B, r_new=5, generator=gen())
    A_new.requires_grad_(True)
    B_new.requires_grad_(True)
    x = torch.randn(4, 8, generator=gen())
    target = torch.randn(4, 6, generator=gen())
    loss = ((x @ A_new.T @ B_new.T) * effective_scaling(5) - target).pow(2).sum()
    loss.backward()
    assert B_new.grad is not None
    assert torch.all(B_new.grad[:, 2:] != 0), "new B columns received zero grad"


def test_a4_new_a_gradient_initially_zero():
    torch.manual_seed(SEED)
    A, B = make_factors(r_old=2, d_in=8, d_out=6)
    A_new, B_new = expand_lora_factors(A, B, r_new=5, generator=gen())
    A_new.requires_grad_(True)
    B_new.requires_grad_(True)
    x = torch.randn(4, 8, generator=gen())
    target = torch.randn(4, 6, generator=gen())
    loss = ((x @ A_new.T @ B_new.T) * effective_scaling(5) - target).pow(2).sum()
    loss.backward()
    assert torch.all(A_new.grad[2:] == 0), "new A rows should receive exactly zero grad while B columns are zero"
    assert torch.all(A_new.grad[:2] != 0), "old A rows should receive nonzero grad"


def test_a5_new_a_gradient_nonzero_after_b_update():
    torch.manual_seed(SEED)
    A, B = make_factors(r_old=2, d_in=8, d_out=6)
    A_new, B_new = expand_lora_factors(A, B, r_new=5, generator=gen())
    A_new.requires_grad_(True)
    B_new.requires_grad_(True)
    x = torch.randn(4, 8, generator=gen())
    target = torch.randn(4, 6, generator=gen())

    def loss_fn():
        return ((x @ A_new.T @ B_new.T) * effective_scaling(5) - target).pow(2).sum()

    opt = torch.optim.AdamW([A_new, B_new], lr=0.05)
    loss_fn().backward()
    opt.step()
    assert torch.all(B_new[:, 2:] != 0), "B update should make new columns nonzero"
    opt.zero_grad(set_to_none=True)
    loss_fn().backward()
    assert torch.all(A_new.grad[2:] != 0), "new A rows should get nonzero grad after B update"


def test_a6_new_components_learn():
    """Synthetic linear regression with a rank-4 target; expanded adapter
    (rank 6, all components trainable) must beat the frozen-original-rank
    baseline (new components frozen at zero == rank-2 capacity)."""
    torch.manual_seed(SEED)
    g = gen(SEED)
    d_in, d_out, n = 16, 8, 256
    # rank-4 target map
    Wt = torch.randn(d_out, 16, generator=g, dtype=torch.float64)
    U, S, Vh = torch.linalg.svd(Wt, full_matrices=False)
    W_target = U[:, :4] @ torch.diag(S[:4]) @ Vh[:4]
    X = torch.randn(n, d_in, generator=g, dtype=torch.float64)
    Y = X @ W_target.T + 0.01 * torch.randn(n, d_out, generator=g, dtype=torch.float64)

    def run(train_new: bool, steps: int = 500, lr: float = 0.05):
        A0, B0 = make_factors(r_old=2, d_in=d_in, d_out=d_out, dtype=torch.float64, seed=SEED + 1)
        A, B = expand_lora_factors(A0, B0, r_new=6, generator=gen(SEED + 2))
        A.requires_grad_(True)
        B.requires_grad_(True)
        if not train_new:  # freeze new components at zero == original-rank capacity
            A_old_only = A[:2].detach().requires_grad_(True)
            B_old_only = B[:, :2].detach().requires_grad_(True)
            params = [A_old_only, B_old_only]

            def forward():
                return (X @ A_old_only.T @ B_old_only.T) * effective_scaling(2)
        else:
            params = [A, B]

            def forward():
                return (X @ A.T @ B.T) * effective_scaling(6)

        opt = torch.optim.AdamW(params, lr=lr)  # fresh optimizer per event
        first = loss = None
        for step in range(steps):
            opt.zero_grad(set_to_none=True)
            loss = (forward() - Y).pow(2).mean()
            if first is None:
                first = loss.item()
            loss.backward()
            opt.step()
        return first, loss.item()

    first_full, final_full = run(train_new=True)
    _, final_baseline = run(train_new=False)
    assert final_full < first_full, "expanded adapter did not learn"
    assert final_full < final_baseline, (
        f"expanded final loss {final_full:.6f} should beat frozen original-rank "
        f"baseline {final_baseline:.6f}"
    )


def test_a7_optimizer_recreation_fresh():
    torch.manual_seed(SEED)
    # Event k: train briefly so the old optimizer holds nonzero moments.
    A_old = torch.randn(2, 8, generator=gen(SEED))
    B_old = torch.randn(6, 2, generator=gen(SEED))
    A_old.requires_grad_(True)
    B_old.requires_grad_(True)
    old_opt = torch.optim.AdamW([A_old, B_old], lr=1e-2)
    x = torch.randn(4, 8, generator=gen())
    y = torch.randn(4, 6, generator=gen())

    def old_loss():
        return ((x @ A_old.T @ B_old.T) * 2.0 - y).pow(2).sum()

    old_loss().backward()
    old_opt.step()
    assert old_opt.state, "old optimizer should hold lazily created state after a step"
    old_state_snapshot = {id(p): (s["exp_avg"].clone(), s["step"].clone()) for p, s in old_opt.state.items()}

    # Event boundary: rank transition produces NEW parameter objects.
    A_new, B_new = expand_lora_factors(A_old.detach(), B_old.detach(), r_new=5, generator=gen())
    assert A_new.data_ptr() != A_old.data_ptr() and B_new.data_ptr() != B_old.data_ptr()
    # Boundary snapshot: old components bitwise preserved at the transition.
    A_boundary = A_new.detach().clone()
    B_boundary = B_new.detach().clone()
    A_new.requires_grad_(True)
    B_new.requires_grad_(True)

    # Fresh AdamW over the new parameters; nothing copied from old_opt.
    new_opt = torch.optim.AdamW([A_new, B_new], lr=1e-2)
    assert new_opt is not old_opt
    new_param_ids = {id(p) for grp in new_opt.param_groups for p in grp["params"]}
    assert new_param_ids == {id(A_new), id(B_new)}
    assert id(A_old) not in new_param_ids and id(B_old) not in new_param_ids
    # PyTorch lazily initializes AdamW state: empty immediately after
    # construction; treat as zero-initialized state, not a failure.
    assert len(new_opt.state) == 0

    # One step with fresh gradients g; step-1 moments must match
    # exp_avg = (1-b1)*g and exp_avg_sq = (1-b2)*g^2 exactly (fresh init),
    # which discriminates against migrated state (old exp_avg was nonzero
    # and would change the recurrence).
    def new_loss():
        return ((x @ A_new.T @ B_new.T) * effective_scaling(5) - y).pow(2).sum()

    new_opt.zero_grad(set_to_none=True)
    new_loss().backward()
    grads = {id(p): p.grad.clone() for p in (A_new, B_new)}
    new_opt.step()
    b1, b2 = new_opt.param_groups[0]["betas"]
    for p in (A_new, B_new):
        s = new_opt.state[p]
        g = grads[id(p)]
        assert int(s["step"].item()) == 1
        assert torch.allclose(s["exp_avg"], (1 - b1) * g, rtol=1e-5, atol=1e-8)
        assert torch.allclose(s["exp_avg_sq"], (1 - b2) * g.pow(2), rtol=1e-5, atol=1e-8)
    # old moments differ from the fresh-recurrence values -> migration would fail this check
    for pid, (old_avg, old_step) in old_state_snapshot.items():
        assert old_step.item() == 1 and old_avg.abs().sum() > 0

    # New factors: old components bitwise preserved at the boundary (before
    # the new event's optimizer steps), so the transition did not alter the
    # adapter beyond the intended rank change. After the step, old rows
    # legitimately continue training under the new event.
    assert torch.equal(A_boundary[:2], A_old.detach())
    assert torch.equal(B_boundary[:, :2], B_old.detach())


# ---------------------------------------------------------------------------
# B-tests: thin/core-SVD reduction
# ---------------------------------------------------------------------------


def observed_rel_err(A: torch.Tensor, B: torch.Tensor, A_new: torch.Tensor, B_new: torch.Tensor) -> float:
    """Independent full-matrix Frobenius error (small test tensors only)."""
    M = (B @ A) * effective_scaling(A.shape[0])
    M_rec = (B_new @ A_new) * effective_scaling(A_new.shape[0])
    denom = torch.linalg.norm(M)
    if denom == 0:
        return 0.0
    return float((torch.linalg.norm(M - M_rec) / denom).item())


def predicted_tail_error(A: torch.Tensor, B: torch.Tensor, r_new: int) -> float:
    """sqrt(sum(discarded S^2)) / sqrt(sum(all S^2)) from full svdvals(B@A)."""
    S = torch.linalg.svdvals((B @ A) * effective_scaling(A.shape[0]))
    total = torch.sum(S**2)
    if not (total > 0):
        return 0.0
    return float(torch.sqrt(torch.sum(S[r_new:] ** 2) / total).item())


def test_b1_reconstruction_matches_tail_prediction():
    A, B = make_factors(r_old=8, d_in=32, d_out=24, dtype=torch.float64, seed=SEED + 3)
    A_new, B_new, rel = reduce_lora_factors(A, B, r_new=4)
    pred = predicted_tail_error(A, B, 4)
    obs = observed_rel_err(A, B, A_new, B_new)
    assert abs(obs - pred) < F64_TAIL, f"observed {obs} vs predicted {pred}"
    assert abs(rel - pred) < F64_TAIL, f"core-exposed rel_err {rel} vs predicted {pred}"


def test_b2_truncation_error_bounded_and_structured():
    # flat-spectrum random reference
    A, B = make_factors(r_old=8, d_in=32, d_out=24, dtype=torch.float64, seed=SEED + 4)
    _, _, rel_flat = reduce_lora_factors(A, B, r_new=4)
    assert 0.0 < rel_flat <= 1.0 + 1e-12

    # deterministic structured spectrum: geometric singular values
    g = gen(SEED + 5)
    U, _ = torch.linalg.qr(torch.randn(24, 8, generator=g, dtype=torch.float64))
    V, _ = torch.linalg.qr(torch.randn(32, 8, generator=g, dtype=torch.float64))
    S = 0.5 ** torch.arange(8, dtype=torch.float64)
    B_s = U @ torch.diag(S)
    A_s = V.T
    _, _, rel_struct = reduce_lora_factors(A_s, B_s, r_new=4)
    assert 0.0 <= rel_struct <= 1.0 + 1e-12
    # recorded reference values (not universal expectations)
    assert rel_struct < rel_flat, "structured decaying spectrum should truncate better than flat"
    # additional random cases all respect the bound
    for seed in range(SEED + 6, SEED + 11):
        A_i, B_i = make_factors(r_old=8, d_in=32, d_out=24, dtype=torch.float64, seed=seed)
        _, _, rel_i = reduce_lora_factors(A_i, B_i, r_new=4)
        assert 0.0 <= rel_i <= 1.0 + 1e-12


def test_b3_zero_norm_adapter():
    A = torch.zeros(4, 16, dtype=torch.float64)
    B = torch.zeros(12, 4, dtype=torch.float64)
    A_new, B_new, rel = reduce_lora_factors(A, B, r_new=2)
    assert torch.all(A_new == 0) and torch.all(B_new == 0)
    assert rel == 0.0
    for t in (A_new, B_new):
        assert torch.isfinite(t).all()
    assert A_new.shape == (2, 16) and B_new.shape == (12, 2)


def test_b4_numerical_tolerance_against_direct_svd():
    A, B = make_factors(r_old=8, d_in=32, d_out=24, dtype=torch.float64, seed=SEED + 12)
    r_new = 4
    A_new, B_new, rel = reduce_lora_factors(A, B, r_new)
    # tail agreement (already in B1; different seed here)
    assert abs(rel - predicted_tail_error(A, B, r_new)) < F64_TAIL

    # direct SVD on the same UNSCALED product B@A (the scaling alpha/r is
    # applied outside the factors, so both paths must be compared on the
    # same unscaled basis); compare RECONSTRUCTED MATRICES and SINGULAR
    # VALUES (singular vectors may have non-unique signs/bases)
    M_unscaled = B @ A
    U, S, Vh = torch.linalg.svd(M_unscaled, full_matrices=False)
    M_direct = (U[:, :r_new] * S[:r_new]) @ Vh[:r_new]
    M_core = B_new @ A_new
    # Hardened assertion (preflight 2026-11-21): the B4 specification is an
    # ABSOLUTE 1e-12 bound, so relative tolerance is disabled (rtol=0.0).
    # With rtol=1e-10 and entries of magnitude O(10), allclose would have
    # admitted deviations up to ~1e-9 — 1000x looser than specified.
    d_mat = float((M_core - M_direct).abs().max().item())
    assert d_mat <= F64_STRICT, (
        f"QR+core-SVD path deviates from direct SVD beyond 1e-12 "
        f"(max abs diff {d_mat:.3e})"
    )
    sv_core = torch.linalg.svdvals(B_new @ A_new)
    d_sv = float((sv_core[:r_new] - S[:r_new]).abs().max().item())
    assert d_sv <= F64_STRICT, (
        f"singular values deviate beyond 1e-12 (max abs diff {d_sv:.3e})"
    )


def test_b5_output_rank_and_shapes():
    A, B = make_factors(r_old=8, d_in=32, d_out=24, dtype=torch.float64, seed=SEED + 13)
    r_new = 3
    A_new, B_new, _ = reduce_lora_factors(A, B, r_new)
    assert A_new.shape == (r_new, 32) and B_new.shape == (24, r_new)
    M_new = B_new @ A_new
    # numerical rank with stated tolerance: tol = S_max * max(m,n) * eps
    S_full = torch.linalg.svdvals(B @ A)  # unscaled basis, matches M_new
    tol = float(S_full.max().item()) * max(M_new.shape) * torch.finfo(torch.float64).eps
    rank = int(torch.linalg.matrix_rank(M_new, tol=tol).item())
    assert rank <= r_new, f"numerical rank {rank} exceeds r_new {r_new}"
    # top r_new singular values are preserved; the discarded tail contributes
    # nothing (svdvals of a rank-r_new matrix still returns min(m,n) values —
    # the trailing ones must be ~zero within the stated tolerance)
    S_new = torch.linalg.svdvals(M_new)
    # Hardened assertion (preflight 2026-11-21): absolute 1e-12 bound only
    # (rtol=0.0); the prior rtol=1e-10 would have allowed deviations up to
    # sigma_max * 1e-10 ~ 1e-8 here, far looser than the specified bound.
    d_sv = float((S_new[:r_new] - S_full[:r_new]).abs().max().item())
    assert d_sv <= F64_STRICT, (
        f"retained singular values deviate beyond 1e-12 (max abs diff {d_sv:.3e})"
    )
    assert torch.all(S_new[r_new:] <= tol), "retained factors leak discarded singular components"
    # note: floating-point multiplication verifies rank only numerically,
    # not exact algebraic rank — asserted via the stated tolerance.


# ---------------------------------------------------------------------------
# alpha convention and invalid-input handling
# ---------------------------------------------------------------------------


def test_alpha_convention_rank_invariant():
    for r in (1, 2, 4, 8, 9, 16, 32):
        assert alpha_for_rank(r) == 2 * r
        assert effective_scaling(r) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        alpha_for_rank(0)
    with pytest.raises(ValueError):
        alpha_for_rank(-2)
    with pytest.raises(ValueError):
        alpha_for_rank(2.0)  # type: ignore[arg-value]


def test_invalid_inputs_rejected():
    A, B = make_factors(r_old=4, d_in=16, d_out=12)
    # expansion
    with pytest.raises(ValueError):
        expand_lora_factors(A, B, r_new=4)  # not an expansion
    with pytest.raises(ValueError):
        expand_lora_factors(A, B, r_new=2)
    with pytest.raises(ValueError):
        expand_lora_factors(A, B, r_new=0)
    # reduction
    with pytest.raises(ValueError):
        reduce_lora_factors(A, B, r_new=4)  # r_new == r_old
    with pytest.raises(ValueError):
        reduce_lora_factors(A, B, r_new=8)  # r_new > r_old
    with pytest.raises(ValueError):
        reduce_lora_factors(A, B, r_new=0)
    with pytest.raises(ValueError):
        reduce_lora_factors(A, B, r_new=-1)
    # shape mismatches
    with pytest.raises(ValueError):
        expand_lora_factors(torch.randn(4, 16), torch.randn(12, 5), r_new=6)
    with pytest.raises(ValueError):
        reduce_lora_factors(torch.randn(4, 16), torch.randn(12, 5), r_new=2)
    # non-2D
    with pytest.raises(ValueError):
        expand_lora_factors(torch.randn(4, 16, 1), torch.randn(12, 4), r_new=6)
    # dtype mismatch / non-floating
    with pytest.raises(ValueError):
        expand_lora_factors(torch.randn(4, 16, dtype=torch.float64), torch.randn(12, 4), r_new=6)
    with pytest.raises(ValueError):
        reduce_lora_factors(torch.randint(0, 5, (4, 16)), torch.randn(12, 4), r_new=2)
    # non-finite input rejected by reduction
    A_nan = A.clone()
    A_nan[0, 0] = float("nan")
    with pytest.raises(ValueError):
        reduce_lora_factors(A_nan, B, r_new=2)


# ---------------------------------------------------------------------------
# Audit corrections (2026-11-21): trainable-input safety, feasibility,
# explicit dtype contract. A1-A7 / B1-B5 are unchanged and still authoritative.
# ---------------------------------------------------------------------------


def test_f1_expansion_trainable_inputs_optimizer_ready():
    """Finding 1: expansion of nn.Parameter inputs yields detached outputs
    that can be wrapped in fresh nn.Parameter objects and driven by a
    fresh AdamW optimizer."""
    torch.manual_seed(SEED)
    A_param = torch.nn.Parameter(make_factors(r_old=2, d_in=8, d_out=6)[0])
    B_param = torch.nn.Parameter(make_factors(r_old=2, d_in=8, d_out=6, seed=SEED + 1)[1])
    A_snapshot = A_param.detach().clone()
    B_snapshot = B_param.detach().clone()

    A_new, B_new = expand_lora_factors(A_param, B_param, r_new=5, generator=gen())
    # no autograd history: plain detached tensors, not non-leaf views
    assert not A_new.requires_grad and A_new.grad_fn is None
    assert not B_new.requires_grad and B_new.grad_fn is None

    # wrap in new Parameter objects and run a fresh AdamW step
    P_A = torch.nn.Parameter(A_new)
    P_B = torch.nn.Parameter(B_new)
    assert P_A.is_leaf and P_B.is_leaf
    opt = torch.optim.AdamW([P_A, P_B], lr=1e-2)  # must construct cleanly
    x = torch.randn(4, 8, generator=gen())
    y = torch.randn(4, 6, generator=gen())
    loss = ((x @ P_A.T @ P_B.T) * effective_scaling(5) - y).pow(2).sum()
    loss.backward()
    opt.step()  # must execute cleanly
    assert P_A.grad is not None and P_B.grad is not None

    # original parameters untouched (values bitwise equal, still trainable)
    assert torch.equal(A_param.detach(), A_snapshot)
    assert torch.equal(B_param.detach(), B_snapshot)
    assert A_param.requires_grad and B_param.requires_grad


def test_f1_reduction_trainable_inputs_optimizer_ready():
    """Finding 1: same checks for SVD reduction."""
    torch.manual_seed(SEED)
    A_param = torch.nn.Parameter(make_factors(r_old=8, d_in=16, d_out=12, dtype=torch.float64)[0])
    B_param = torch.nn.Parameter(make_factors(r_old=8, d_in=16, d_out=12, dtype=torch.float64, seed=SEED + 1)[1])
    A_snapshot = A_param.detach().clone()
    B_snapshot = B_param.detach().clone()

    A_new, B_new, rel = reduce_lora_factors(A_param, B_param, r_new=4)
    assert not A_new.requires_grad and A_new.grad_fn is None
    assert not B_new.requires_grad and B_new.grad_fn is None
    assert 0.0 <= rel <= 1.0

    P_A = torch.nn.Parameter(A_new)
    P_B = torch.nn.Parameter(B_new)
    assert P_A.is_leaf and P_B.is_leaf
    opt = torch.optim.AdamW([P_A, P_B], lr=1e-2)
    x = torch.randn(4, 16, generator=gen(), dtype=torch.float64)
    y = torch.randn(4, 12, generator=gen(), dtype=torch.float64)
    loss = ((x @ P_A.T @ P_B.T) * effective_scaling(4) - y).pow(2).sum()
    loss.backward()
    opt.step()
    assert P_A.grad is not None and P_B.grad is not None

    assert torch.equal(A_param.detach(), A_snapshot)
    assert torch.equal(B_param.detach(), B_snapshot)
    assert A_param.requires_grad and B_param.requires_grad


def test_f2_reduction_rank_boundary_feasible():
    """Finding 2: r_new == min(d_out, d_in) is accepted; reductions remain
    valid even when r_old exceeds the maximum matrix rank."""
    # boundary: min(d_out=6, d_in=4) == 4 < r_old == 8 -> feasible
    A, B = make_factors(r_old=8, d_in=4, d_out=6, dtype=torch.float64, seed=SEED + 20)
    A_new, B_new, rel = reduce_lora_factors(A, B, r_new=4)
    assert A_new.shape == (4, 4) and B_new.shape == (6, 4)
    assert abs(rel - predicted_tail_error(A, B, 4)) < F64_TAIL

    # degenerate original rank r_old=10 > min(d_out=6, d_in=4)=4, feasible
    # r_new=3 -> must still work (finding 2 preservation requirement)
    A_d, B_d = make_factors(r_old=10, d_in=4, d_out=6, dtype=torch.float64, seed=SEED + 21)
    A_dn, B_dn, rel_d = reduce_lora_factors(A_d, B_d, r_new=3)
    assert A_dn.shape == (3, 4) and B_dn.shape == (6, 3)
    assert abs(rel_d - predicted_tail_error(A_d, B_d, 3)) < F64_TAIL


def test_f2_reduction_infeasible_rank_rejected():
    """Finding 2: r_new > min(d_out, d_in) raises ValueError."""
    A, B = make_factors(r_old=8, d_in=4, d_out=6, dtype=torch.float64, seed=SEED + 22)
    with pytest.raises(ValueError, match="min\\(d_out, d_in\\)"):
        reduce_lora_factors(A, B, r_new=5)   # just above the bound
    with pytest.raises(ValueError, match="infeasible"):
        reduce_lora_factors(A, B, r_new=7)   # far above the bound, still < r_old
    # also when r_old itself exceeds the bound and r_new sits between
    A_d, B_d = make_factors(r_old=10, d_in=4, d_out=6, dtype=torch.float64, seed=SEED + 23)
    with pytest.raises(ValueError):
        reduce_lora_factors(A_d, B_d, r_new=6)  # 6 > min(6, 4) = 4


def test_f3_dtype_contract_explicit():
    """Finding 3: float32/float64 supported with dtype preserved;
    FP16/BF16 rejected with a clear ValueError, no implicit conversion."""
    A32, B32 = make_factors(r_old=4, d_in=16, d_out=12, dtype=torch.float32)
    A32n, B32n = expand_lora_factors(A32, B32, r_new=8)
    assert A32n.dtype == torch.float32 and B32n.dtype == torch.float32
    A32r, B32r, _ = reduce_lora_factors(A32, B32, r_new=2)
    assert A32r.dtype == torch.float32 and B32r.dtype == torch.float32

    A64, B64 = make_factors(r_old=4, d_in=16, d_out=12, dtype=torch.float64)
    A64n, B64n = expand_lora_factors(A64, B64, r_new=8)
    assert A64n.dtype == torch.float64 and B64n.dtype == torch.float64

    for dtype in (torch.float16, torch.bfloat16):
        A_h = torch.randn(4, 16, generator=gen(), dtype=dtype)
        B_h = torch.randn(12, 4, generator=gen(), dtype=dtype)
        with pytest.raises(ValueError, match=r"torch\.float32 and torch\.float64"):
            expand_lora_factors(A_h, B_h, r_new=6)
        with pytest.raises(ValueError, match=r"torch\.float32 and torch\.float64"):
            reduce_lora_factors(A_h, B_h, r_new=2)
