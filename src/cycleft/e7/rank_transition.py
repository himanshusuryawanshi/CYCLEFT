"""E7 Phase 1: LoRA rank-transition mathematics (CPU-only, pure torch).

Conventions (confirmed against PEFT 0.21.2 ``LoraLayer.update_layer``,
non-rslora default ``scaling = lora_alpha / r``):

- ``A`` has shape ``[rank, input_dim]``   (PEFT ``lora_A.weight``)
- ``B`` has shape ``[output_dim, rank]``  (PEFT ``lora_B.weight``)
- Effective adapter update: ``E = (B @ A) * (alpha / rank)`` with the
  E7 protocol ``alpha = 2 * rank``, so the effective scaling is the
  constant 2 and is invariant under rank changes.

Expansion preserves the existing factors bitwise and appends Kaiming-A /
zero-B components, so the effective update is unchanged at the boundary.
Reduction uses the thin/core-SVD algorithm from the v5 spec
(EXPANSION_AND_SVD_TEST_SPEC.md); its relative Frobenius error is
computed from the core singular values without materializing the full
``B @ A`` matrix (the QR isometries preserve Frobenius norms, so
``||B @ A||_F == ||R_B @ R_A^T||_F`` and the truncation error equals the
tail-energy of the core spectrum exactly).

This module implements no optimizer migration: per
OPTIMIZER_LIFECYCLE_FINAL.md, event boundaries discard the old AdamW and
construct a fresh one; nothing here copies or transforms optimizer state.

Contract (audit corrections, 2026-11-21):
- Transitions are checkpoint/parameter transformations: inputs are read
  through detached views, outputs are plain tensors with no autograd
  history, safe to wrap in ``torch.nn.Parameter`` for a fresh optimizer.
- Supported dtypes: torch.float32 and torch.float64 only (FP16/BF16 are
  rejected with ValueError; no implicit conversion). Output dtype always
  equals input dtype.
- Reduction requires ``r_new <= min(d_out, d_in)`` (infeasible ranks
  raise ValueError); ``r_old`` above that bound remains reducible.
"""

from __future__ import annotations

import math

import torch
import torch.nn.init as _init

__all__ = ["alpha_for_rank", "effective_scaling", "expand_lora_factors", "reduce_lora_factors"]


def alpha_for_rank(rank: int) -> int:
    """E7 protocol alpha: ``alpha = 2 * rank``."""
    _validate_rank(rank, "rank")
    return 2 * rank


def effective_scaling(rank: int) -> float:
    """Effective LoRA scaling ``alpha / rank`` — invariantly 2.0 under the
    E7 ``alpha = 2 * rank`` protocol, for every rank."""
    return alpha_for_rank(rank) / rank


def _validate_rank(rank: int, name: str) -> None:
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise ValueError(f"{name} must be an int, got {type(rank).__name__}")
    if rank <= 0:
        raise ValueError(f"{name} must be positive, got {rank}")


def _validate_factors(A: torch.Tensor, B: torch.Tensor) -> None:
    if A.dim() != 2 or B.dim() != 2:
        raise ValueError(
            f"A and B must be 2-D [rank, d_in] and [d_out, rank]; got "
            f"A{tuple(A.shape)}, B{tuple(B.shape)}"
        )
    if not (A.is_floating_point() and B.is_floating_point()):
        raise ValueError(f"A and B must be floating-point; got {A.dtype} and {B.dtype}")
    # Explicit dtype contract (audit correction, Finding 3): CPU Phase 1
    # supports float32 and float64 only. FP16/BF16 are rejected rather than
    # silently converted; the E7 protocol runs LoRA factors in fp32 anyway
    # (fp16 base + fp32 LoRA). Output dtype always equals input dtype.
    if A.dtype not in (torch.float32, torch.float64):
        raise ValueError(
            f"unsupported dtype {A.dtype}: rank transitions support "
            f"torch.float32 and torch.float64 only; convert explicitly "
            f"before calling (no implicit cast is performed)"
        )
    if A.dtype != B.dtype:
        raise ValueError(f"A and B must share a dtype; got {A.dtype} and {B.dtype}")
    if A.device != B.device:
        raise ValueError(f"A and B must share a device; got {A.device} and {B.device}")
    if A.shape[0] != B.shape[1]:
        raise ValueError(
            f"rank mismatch: A is [r, d_in] with r={A.shape[0]} but B is "
            f"[d_out, r] with r={B.shape[1]}"
        )
    if A.shape[0] == 0:
        raise ValueError("rank must be positive; got r=0")


def expand_lora_factors(
    A: torch.Tensor,
    B: torch.Tensor,
    r_new: int,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-preserving rank expansion ``r_old -> r_new`` (``r_new > r_old``).

    Returns NEW factor tensors; the inputs are not modified.

    - Existing A rows ``[:r_old]`` and B columns ``[:, :r_old]`` are
      preserved bitwise (copied verbatim, same dtype/device).
    - New A rows are initialized with ``torch.nn.init.kaiming_uniform_``
      with ``a=math.sqrt(5)`` (the PEFT default) on the ``[r_new - r_old,
      d_in]`` region.
    - New B columns are exactly zero, so ``B_new @ A_new == B @ A`` and
      the effective update is unchanged at the boundary.

    Rank transitions are checkpoint/parameter transformations, not
    differentiable operations (audit correction, Finding 1): inputs are
    read via detached views, so the returned tensors never carry autograd
    history and can be wrapped in ``torch.nn.Parameter`` and used in a
    fresh optimizer directly. Original inputs are never modified.

    The caller keeps ``alpha = 2 * rank`` (see :func:`alpha_for_rank`);
    because ``alpha / rank == 2`` for every rank, no rescaling of the
    existing factors is required or performed.
    """
    _validate_factors(A, B)
    _validate_rank(r_new, "r_new")
    r_old = A.shape[0]
    if r_new <= r_old:
        raise ValueError(f"expansion requires r_new > r_old ({r_new} <= {r_old})")
    A, B = A.detach(), B.detach()  # no autograd graph on outputs

    d_in, d_out = A.shape[1], B.shape[0]
    A_new = torch.empty((r_new, d_in), dtype=A.dtype, device=A.device)
    B_new = torch.empty((d_out, r_new), dtype=B.dtype, device=B.device)

    A_new[:r_old].copy_(A)
    B_new[:, :r_old].copy_(B)
    _init.kaiming_uniform_(A_new[r_old:], a=math.sqrt(5), generator=generator)
    B_new[:, r_old:].zero_()

    return A_new, B_new


def reduce_lora_factors(
    A: torch.Tensor,
    B: torch.Tensor,
    r_new: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Thin/core-SVD rank reduction ``r_old -> r_new`` (``0 < r_new < r_old``).

    Algorithm (v5 spec, unchanged):

        M       = B @ A                       # conceptually [d_out, d_in]
        Q_B,R_B = qr(B)                       # Q_B [d_out, r], R_B [r, r]
        Q_A,R_A = qr(A.T)                     # Q_A [d_in, r],  R_A [r, r]
        U_c,S_c,Vh_c = svd(R_B @ R_A.T)       # [r, r] core
        U   = Q_B @ U_c
        Vh  = Vh_c @ Q_A.T
        A_new = S_c[:r_new, None] * Vh[:r_new, :]
        B_new = U[:, :r_new]

    The full matrix ``M`` is never materialized: the QR isometries
    preserve Frobenius norms, so the relative truncation error equals the
    discarded tail energy of the ``r x r`` core spectrum,

        rel_err = sqrt(sum(S_c[r_new:]**2)) / sqrt(sum(S_c**2)),

    which by the Eckart-Young theorem is the optimal rank-``r_new``
    Frobenius approximation error and is <= 1. A zero-norm adapter (all
    singular values zero) returns zero factors with ``rel_err = 0.0``.

    Feasibility (audit correction, Finding 2): the effective update is an
    ``[d_out, d_in]`` matrix whose rank can never exceed
    ``min(d_out, d_in)``; an ``r_new`` above that bound raises
    ``ValueError`` instead of silently returning fewer components.
    Reductions remain valid when ``r_old`` itself exceeds that bound, as
    long as the requested ``r_new`` is feasible.

    Rank transitions are checkpoint/parameter transformations, not
    differentiable operations (audit correction, Finding 1): inputs are
    read via detached views, so the returned tensors never carry autograd
    history and can be wrapped in ``torch.nn.Parameter`` and used in a
    fresh optimizer directly. Original inputs are never modified.

    Returns ``(A_new, B_new, rel_err)``.
    """
    _validate_factors(A, B)
    _validate_rank(r_new, "r_new")
    r_old = A.shape[0]
    if r_new >= r_old:
        raise ValueError(f"reduction requires 0 < r_new < r_old ({r_new} >= {r_old})")
    max_rank = min(A.shape[1], B.shape[0])
    if r_new > max_rank:
        raise ValueError(
            f"infeasible reduction: r_new={r_new} exceeds min(d_out, d_in)="
            f"{max_rank} for an output matrix of shape "
            f"[{B.shape[0]}, {A.shape[1]}]; no rank-{r_new} matrix exists "
            f"in that shape"
        )
    A, B = A.detach(), B.detach()  # no autograd graph on outputs
    for name, t in (("A", A), ("B", B)):
        if not torch.isfinite(t).all():
            raise ValueError(f"{name} contains NaN or Inf")

    Q_B, R_B = torch.linalg.qr(B, mode="reduced")
    Q_A, R_A = torch.linalg.qr(A.T, mode="reduced")
    U_c, S_c, Vh_c = torch.linalg.svd(R_B @ R_A.T, full_matrices=False)

    total_energy = torch.sum(S_c**2)
    if not (total_energy > 0):  # zero-norm adapter
        A_new = torch.zeros((r_new, A.shape[1]), dtype=A.dtype, device=A.device)
        B_new = torch.zeros((B.shape[0], r_new), dtype=B.dtype, device=B.device)
        return A_new, B_new, 0.0

    tail_energy = torch.sum(S_c[r_new:] ** 2)
    rel_err = float(torch.sqrt(tail_energy / total_energy).item())

    U = Q_B @ U_c
    Vh = Vh_c @ Q_A.T
    A_new = S_c[:r_new, None] * Vh[:r_new, :]
    B_new = U[:, :r_new]

    return A_new.contiguous(), B_new.contiguous(), rel_err
