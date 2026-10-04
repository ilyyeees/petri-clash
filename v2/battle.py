"""Headless, symmetric territory rules for two independently trained NCAs.

Ownership is a combat boundary, not a growth mask. Neutral cells keep each
organism's hidden state while they mature; only an opponent's owned cells are
excluded. This distinction is essential for pretrained growing NCAs, whose
frontier usually needs several updates before it becomes visibly alive.
"""

import math

import torch
import torch.nn.functional as F


DEFAULT_PRESSURE_GAIN = 0.28
DEFAULT_CONTROL_DECAY = 0.97
DEFAULT_CAPTURE_THRESHOLD = 0.55
DEFAULT_RELEASE_THRESHOLD = 0.12
DEFAULT_TIE_MARGIN = 0.01
CLAIM_ALPHA = 0.05
ALIVE_ALPHA = 0.1


def validate_rules(pressure_gain, control_decay, capture_threshold, release_threshold, tie_margin):
    """Reject settings that invert the battle or make ownership unreachable."""
    values = (pressure_gain, control_decay, capture_threshold, release_threshold, tie_margin)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("battle settings must be finite")
    if pressure_gain < 0:
        raise ValueError("pressure_gain must be nonnegative")
    if not 0 <= control_decay <= 1:
        raise ValueError("control_decay must be between 0 and 1")
    if not 0 <= release_threshold < capture_threshold <= 1:
        raise ValueError("thresholds must satisfy 0 <= release < capture <= 1")
    if not 0 <= tie_margin <= 1:
        raise ValueError("tie_margin must be between 0 and 1")


def _check_shapes(state_a, state_b, owner, control):
    if state_a.ndim != 4 or state_b.ndim != 4 or min(state_a.shape[1], state_b.shape[1]) < 4:
        raise ValueError("organism states must have shape [batch, channels >= 4, height, width]")
    expected = (state_a.shape[0], 1, *state_a.shape[-2:])
    if state_b.shape[0] != expected[0] or state_b.shape[-2:] != expected[-2:]:
        raise ValueError("organism states must share their batch and grid dimensions")
    if tuple(owner.shape) != expected or tuple(control.shape) != expected:
        raise ValueError("owner and control must have shape [batch, 1, height, width]")
    if len({state_a.device, state_b.device, owner.device, control.device}) != 1:
        raise ValueError("battle tensors must be on the same device")
    if not state_a.is_floating_point() or not state_b.is_floating_point() or not control.is_floating_point():
        raise ValueError("organism states and control must be floating-point tensors")
    if owner.is_floating_point() or owner.is_complex() or owner.dtype == torch.bool:
        raise ValueError("owner must be an integer tensor with labels 0, 1, or 2")


def _near(mask):
    return F.max_pool2d(mask.to(torch.float32), 3, stride=1, padding=1) > 0


def _canonical_cpu_grid(value):
    """Pass fusion is safe for the app's exact dense CPU layout only."""
    if value.device.type != "cpu" or value.ndim != 4 or torch.compiler.is_compiling():
        return False
    batch, channels, height, width = value.shape
    return (batch > 0 and channels > 0 and height > 0 and width > 0
            and value.stride() == (channels * height * width, height * width, width, 1))


def _finite_cells(state, strict=False, keep=None):
    # An unstable model must not poison neighboring cells on the next update.
    # Preserve every finite hidden value; clipping hidden channels changes the
    # pretrained dynamics. A cell with any nonfinite channel is removed instead.
    if (state.device.type == "cpu" and state.shape[1] > 0
            and not torch.compiler.is_compiling()
            and state.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)):
        # abs/amax cannot overflow finite real values; any NaN or infinity
        # makes the reduction nonfinite. Only the predicate is reduced:
        # surviving state values are returned without arithmetic or clipping.
        finite = torch.isfinite(state.abs().amax(dim=1, keepdim=True))
    else:
        finite = torch.isfinite(state).all(dim=1, keepdim=True)
    if strict and not bool(finite.all()):
        raise FloatingPointError("Non-finite NCA state or proposal")
    # Combine per-cell exclusions before touching every state channel.
    # Strict validation still sees all cells, even ones about to be excluded.
    if keep is not None:
        # Fusing arbitrary strides can change the model's input layout and
        # rand_like's value order. Preserve the original two passes outside
        # exact dense eager CPU grids, including unusual singleton strides.
        if not (_canonical_cpu_grid(state) and _canonical_cpu_grid(keep)):
            return torch.where(keep, torch.where(finite, state, 0.0), 0.0)
        finite = keep & finite
    return torch.where(finite, state, 0.0)


def momentum_owner(control, owner, capture_threshold, release_threshold):
    """Apply sign-aware hysteresis without favoring either team.

    Release is directional. A strong hit can jump past zero in a single tick;
    checking only ``abs(control)`` would incorrectly retain the losing owner.
    """
    next_owner = torch.where(
        ((owner == 1) & (control <= release_threshold))
        | ((owner == 2) & (control >= -release_threshold)),
        0,
        owner,
    )
    next_owner = torch.where(control >= capture_threshold, 1, next_owner)
    return torch.where(control <= -capture_threshold, 2, next_owner)


def clash_step(
    state_a,
    state_b,
    owner,
    control,
    model_a,
    model_b,
    pressure_gain=DEFAULT_PRESSURE_GAIN,
    control_decay=DEFAULT_CONTROL_DECAY,
    capture_threshold=DEFAULT_CAPTURE_THRESHOLD,
    release_threshold=DEFAULT_RELEASE_THRESHOLD,
    tie_margin=DEFAULT_TIE_MARGIN,
    strict_finite=False,
):
    """Advance one battle tick without mutating any input.

    Both proposals are computed before combat resolves. Neutral frontier state
    survives even below the alpha and capture thresholds. Enemy territory may
    receive one-step attack pressure from an adjacent living organism, but no
    enemy hidden state survives there until it is released or captured.

    A claimant must be adjacent to its previous living state or territory. This
    permits growth ahead of slow ownership while excluding distant spontaneous
    claims. Dead tissue and its control are pruned after resolving ownership.
    ``strict_finite`` is an opt-in duel guard: non-finite input/proposals raise
    FloatingPointError before quarantine can conceal an invalid round. The
    default sandbox still quarantines unstable cells without changing its rules.
    """
    validate_rules(pressure_gain, control_decay, capture_threshold, release_threshold, tie_margin)
    _check_shapes(state_a, state_b, owner, control)
    with torch.inference_mode():
        if strict_finite and not bool(torch.isfinite(control).all()):
            raise FloatingPointError("Non-finite battle control")
        owned_a, owned_b = owner == 1, owner == 2
        clean_a = _finite_cells(state_a, strict_finite, ~owned_b)
        clean_b = _finite_cells(state_b, strict_finite, ~owned_a)
        support_a = _near((clean_a[:, 3:4] > ALIVE_ALPHA) | owned_a)
        support_b = _near((clean_b[:, 3:4] > ALIVE_ALPHA) | owned_b)

        proposed_a = model_a(clean_a, steps=1)
        proposed_b = model_b(clean_b, steps=1)
        if proposed_a.shape != state_a.shape or proposed_b.shape != state_b.shape:
            raise ValueError("each model must preserve its organism state's shape")
        proposed_a = _finite_cells(proposed_a, strict_finite, support_a)
        proposed_b = _finite_cells(proposed_b, strict_finite, support_b)
        alpha_a = proposed_a[:, 3:4].clamp(0.0, 1.0)
        alpha_b = proposed_b[:, 3:4].clamp(0.0, 1.0)
        strength_a = torch.where(alpha_a > CLAIM_ALPHA, alpha_a, 0.0)
        strength_b = torch.where(alpha_b > CLAIM_ALPHA, alpha_b, 0.0)
        pressure = strength_a - strength_b
        pressure = torch.where(pressure.abs() >= tie_margin, pressure, 0.0)

        clean_control = torch.nan_to_num(control, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        next_control = (clean_control * control_decay + pressure * pressure_gain).clamp(-1.0, 1.0)
        next_owner = momentum_owner(next_control, owner, capture_threshold, release_threshold)

        # The important rule: neutral cells retain hidden developmental state.
        if (_canonical_cpu_grid(proposed_a) and _canonical_cpu_grid(proposed_b)
                and _canonical_cpu_grid(next_owner)):
            # Life needs only alpha. Combine the small territory/life masks
            # before touching every channel on the common dense CPU path.
            keep_a, keep_b = next_owner != 2, next_owner != 1
            life_a = _near((proposed_a[:, 3:4] > ALIVE_ALPHA) & keep_a)
            life_b = _near((proposed_b[:, 3:4] > ALIVE_ALPHA) & keep_b)
            next_a = torch.where(life_a & keep_a, proposed_a, 0.0)
            next_b = torch.where(life_b & keep_b, proposed_b, 0.0)
        else:
            # The intermediate where also determines layout. Retain it for
            # other layouts/devices rather than changing observable strides.
            next_a = torch.where(next_owner != 2, proposed_a, 0.0)
            next_b = torch.where(next_owner != 1, proposed_b, 0.0)
            life_a = _near(next_a[:, 3:4] > ALIVE_ALPHA)
            life_b = _near(next_b[:, 3:4] > ALIVE_ALPHA)
            next_a = torch.where(life_a, next_a, 0.0)
            next_b = torch.where(life_b, next_b, 0.0)
        abandoned = ((next_owner == 1) & ~life_a) | ((next_owner == 2) & ~life_b)
        next_owner = torch.where(abandoned, 0, next_owner)
        next_control = torch.where(abandoned | ~(life_a | life_b), 0.0, next_control)

    return next_a, next_b, next_owner, next_control


def crater(state_a, state_b, owner, control, gx, gy, radius):
    """Erase a circular area, including latent tissue and momentum, without wrap."""
    _check_shapes(state_a, state_b, owner, control)
    if not all(math.isfinite(float(value)) for value in (gx, gy, radius)) or radius < 0:
        raise ValueError("crater center must be finite and radius must be finite and nonnegative")
    height, width = state_a.shape[-2:]
    yy = torch.arange(height, device=state_a.device).view(1, 1, height, 1)
    xx = torch.arange(width, device=state_a.device).view(1, 1, 1, width)
    mask = (xx - gx).square() + (yy - gy).square() <= radius * radius
    return tuple(torch.where(mask, torch.zeros_like(value), value) for value in (state_a, state_b, owner, control))


def compose_rgba(state_a, state_b, owner, show_frontier=True):
    """Render owned tissue plus a symmetric blend of unclaimed living frontier.

    Neutral mixing has no painter's-order advantage: swap the organisms and
    ownership labels and the image is unchanged. Hidden channels stay hidden.
    """
    rgba_a = torch.nan_to_num(state_a[:, :4], nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
    rgba_b = torch.nan_to_num(state_b[:, :4], nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
    if show_frontier:
        alpha_a, alpha_b = rgba_a[:, 3:4], rgba_b[:, 3:4]
        alpha = 1 - (1 - alpha_a) * (1 - alpha_b)
        rgb = (rgba_a[:, :3] * alpha_a + rgba_b[:, :3] * alpha_b) / (alpha_a + alpha_b).clamp_min(1e-8)
        neutral = torch.cat((rgb, alpha), dim=1)
    else:
        neutral = torch.zeros_like(rgba_a)
    return torch.where(owner == 1, rgba_a, torch.where(owner == 2, rgba_b, neutral))
