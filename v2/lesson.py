"""Deterministic, tick-based rules for an optional solo regrowth lesson.

The caller grows one culture, supplies every post-step living-cell count, and
pauses at each action gate. This module never advances a simulation or consumes
random numbers. A completed lesson means the living-cell count met and held its
target; it does not establish shape fidelity, combat strength, or model quality.
"""

from dataclasses import dataclass, replace
from math import isqrt
from numbers import Integral

import numpy as np


def _integer(value, name, minimum=0):
    """Accept integer counts without truncating floats or accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class LessonConfig:
    """Simulation-tick horizons and an exact rational living-count target."""

    grow_ticks: int = 160
    recovery_ticks: int = 160
    target_num: int = 9
    target_den: int = 10
    hold_ticks: int = 24

    def __post_init__(self):
        for name in ("grow_ticks", "recovery_ticks", "target_num", "target_den", "hold_ticks"):
            object.__setattr__(self, name, _integer(getattr(self, name), name, 1))
        if self.target_num > self.target_den:
            raise ValueError("target_num must not exceed target_den")
        if self.hold_ticks > self.recovery_ticks:
            raise ValueError("hold_ticks must not exceed recovery_ticks")


@dataclass(frozen=True)
class CraterPlan:
    """One inclusive integer-radius cut removing 25–60% of living cells.

    Coordinates are zero-based: x is a column and y is a row. The caller must
    apply this exact circle to all simulation channels, then verify the actual
    living count with ``Lesson.apply_damage`` before allowing recovery.
    """

    x: int
    y: int
    radius: int
    baseline: int
    removed: int

    def __post_init__(self):
        for name in ("x", "y", "radius", "baseline", "removed"):
            minimum = 0 if name in ("x", "y") else 1
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum))
        if not (0 < self.removed < self.baseline
                and 4 * self.removed >= self.baseline
                and 5 * self.removed <= 3 * self.baseline):
            raise ValueError("crater must remove 25–60% of living cells without erasing all of them")

    @property
    def remaining(self):
        return self.baseline - self.removed


def _round_centroid(total, count):
    """Round a nonnegative rational to nearest integer, ties to even."""
    lower, remainder = divmod(total, count)
    return lower + int(2 * remainder > count or (2 * remainder == count and lower % 2))


def choose_crater(living_mask):
    """Choose the smallest safe radius at the rounded living-cell centroid.

    Accept a two-dimensional boolean NumPy array, with rows y and columns x.
    Try integer radii from 1 upward using the inclusive squared-distance circle
    used by the arena. Rounding is exact, ties to even; fraction bounds use
    integer comparisons. No input or RNG state is changed. Return None for an
    empty population or if a radius jumps past the safe range. Never relocate
    the center or substitute a different damage rule.
    """
    if not isinstance(living_mask, np.ndarray):
        raise TypeError("living_mask must be a NumPy array")
    if living_mask.ndim != 2 or living_mask.dtype != np.dtype(bool):
        raise ValueError("living_mask must be a two-dimensional boolean array")
    ys, xs = np.nonzero(living_mask)
    baseline = int(xs.size)
    if not baseline:
        return None
    x = _round_centroid(sum(map(int, xs)), baseline)
    y = _round_centroid(sum(map(int, ys)), baseline)
    distances = (xs - x) ** 2 + (ys - y) ** 2
    # Cover every living cell at the latest, without a wasteful unbounded scan.
    farthest = int(distances.max())
    max_radius = max(1, isqrt(farthest) + int(isqrt(farthest) ** 2 < farthest))
    for radius in range(1, max_radius + 1):
        removed = int(np.count_nonzero(distances <= radius * radius))
        if 5 * removed > 3 * baseline:
            return None
        if 0 < removed < baseline and 4 * removed >= baseline:
            return CraterPlan(x, y, radius, baseline, removed)
    return None


@dataclass(frozen=True)
class _LessonState:
    phase: str = "seed"
    tick: int = 0
    current: int | None = None
    baseline: int | None = None
    plan: CraterPlan | None = None
    remaining_after_cut: int | None = None
    hold: int = 0
    reason: str | None = None


class Lesson:
    """Track growth, one verified cut, and a held living-cell recovery target.

    Every post-step tick must be recorded exactly once, beginning at 1. Growth
    stops at ``grow_ticks``; recovery ticks continue that global sequence only
    after a planned cut and the explicit ``watch`` action. ``clip_steps`` caps
    each batch at the next horizon and returns zero at action gates. Callers
    must also stop a batch when ``can_step`` becomes false, since completion or
    invalidation can happen before the horizon.

    Bad arguments and out-of-order actions raise without mutation. An explicit
    ``finite=False`` instead consumes the attempted tick, ignores its count,
    and freezes an invalid observation. All terminal states stay frozen until
    reset. The controller stores no model, tensor, seed, or random state; the
    caller must reset its simulation separately.
    """

    _TERMINAL = frozenset(("complete", "timeout", "unavailable", "invalid"))

    def __init__(self, config=None, board_cells=None):
        if config is None:
            config = LessonConfig()
        if not isinstance(config, LessonConfig):
            raise TypeError("config must be a LessonConfig")
        if board_cells is not None:
            board_cells = _integer(board_cells, "board_cells", 1)
        self._config = config
        self._board_cells = board_cells
        self.reset()

    @property
    def config(self):
        return self._config

    @property
    def board_cells(self):
        return self._board_cells

    @property
    def phase(self):
        return self._state.phase

    @property
    def tick(self):
        return self._state.tick

    @property
    def can_step(self):
        return self.phase in ("grow", "recover")

    @property
    def finished(self):
        return self.phase in self._TERMINAL

    @property
    def result(self):
        """Return a fresh terminal snapshot, or None before the observation ends."""
        return self.snapshot() if self.finished else None

    def reset(self):
        """Return to the seed gate, keeping the immutable rules and capacity."""
        self._state = _LessonState()
        return self.snapshot()

    def _require_phase(self, phase):
        if self.phase != phase:
            raise ValueError(f"action requires {phase} phase, got {self.phase}")

    def _count(self, living, name="living"):
        living = _integer(living, name)
        if self.board_cells is not None and living > self.board_cells:
            raise ValueError(f"{name} exceeds board_cells")
        return living

    def start_seed(self):
        """Authorize growth after the caller plants the single guided seed."""
        self._require_phase("seed")
        self._state = replace(self._state, phase="grow")
        return self.snapshot()

    def set_crater(self, plan):
        """Attach one safe preview at the growth boundary, or mark unavailable."""
        self._require_phase("damage")
        if self._state.plan is not None:
            raise ValueError("crater preview is already set")
        if plan is None:
            self._state = replace(
                self._state, phase="unavailable",
                reason="No safe practice cut for this state. Try another seed or culture.")
        else:
            if not isinstance(plan, CraterPlan):
                raise TypeError("plan must be a CraterPlan or None")
            if plan.baseline != self._state.baseline:
                raise ValueError("crater baseline does not match the frozen living count")
            self._state = replace(self._state, plan=plan)
        return self.snapshot()

    def apply_damage(self, actual_remaining):
        """Verify the exact preview's result, then pause on the injured state.

        A mismatch raises ValueError without mutating the controller. If the
        caller has already changed its simulation, it must invalidate the
        observation rather than retrying recovery with unverified damage.
        """
        self._require_phase("damage")
        if self._state.plan is None:
            raise ValueError("a crater preview must be set before applying damage")
        remaining = self._count(actual_remaining, "actual_remaining")
        if remaining != self._state.plan.remaining:
            raise ValueError("actual remaining count does not match the crater preview")
        self._state = replace(self._state, phase="injured", current=remaining,
                              remaining_after_cut=remaining)
        return self.snapshot()

    def watch(self):
        """Allow recovery ticks after the injured state has been inspected."""
        self._require_phase("injured")
        self._state = replace(self._state, phase="recover")
        return self.snapshot()

    def clip_steps(self, requested):
        """Return a batch bounded by the next action gate or recovery horizon."""
        requested = _integer(requested, "requested")
        if not self.can_step:
            return 0
        horizon = self.config.grow_ticks
        if self.phase == "recover":
            horizon += self.config.recovery_ticks
        return min(requested, horizon - self.tick)

    def invalidate(self, reason):
        """Freeze a nonterminal observation when its simulation cannot be trusted."""
        if self.finished:
            raise ValueError("lesson is finished; reset before invalidating it")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a nonempty string")
        self._state = replace(self._state, phase="invalid", current=None, reason=reason)
        return self.snapshot()

    def record_tick(self, tick, living, *, finite=True):
        """Record one consecutive post-step count, never a rendered-frame sample."""
        if not self.can_step:
            raise ValueError(f"cannot record a tick during {self.phase} phase")
        tick = _integer(tick, "tick", 1)
        if tick != self.tick + 1:
            raise ValueError(f"expected tick {self.tick + 1}, got {tick}; every tick is required")
        if not isinstance(finite, bool):
            raise ValueError("finite must be a boolean")
        if not finite:
            self._state = replace(
                self._state, phase="invalid", tick=tick, current=None,
                reason="Non-finite simulation state; this observation has no valid result.")
            return self.snapshot()
        living = self._count(living)
        if self.phase == "grow":
            at_boundary = tick == self.config.grow_ticks
            self._state = replace(self._state, tick=tick, current=living,
                                  phase="damage" if at_boundary else "grow",
                                  baseline=living if at_boundary else None)
        else:
            qualifies = self.config.target_den * living >= self.config.target_num * self._state.baseline
            hold = self._state.hold + 1 if qualifies else 0
            phase, reason = "recover", None
            if hold >= self.config.hold_ticks:
                phase = "complete"
            elif tick == self.config.grow_ticks + self.config.recovery_ticks:
                phase = "timeout"
                reason = (f"Observation ended after {self.config.recovery_ticks} recovery ticks. "
                          "The living-cell target was not held.")
            self._state = replace(self._state, tick=tick, current=living,
                                  hold=hold, phase=phase, reason=reason)
        return self.snapshot()

    def snapshot(self):
        """Return independent, JSON-safe progress and exact living-cell counts.

        ``removed`` describes the preview once set; ``remaining_after_cut`` is
        None until that cut has been verified. ``current`` is unknown before
        the first recorded tick and after invalidation. ``valid`` means the
        observation was not invalidated, not that its target was reached.
        Growth/recovery tick counters include an attempted non-finite tick.
        """
        state, config = self._state, self.config
        growth_ticks = min(state.tick, config.grow_ticks)
        recovery_ticks = max(0, state.tick - config.grow_ticks)
        plan = state.plan
        target = None if state.baseline is None else (
            config.target_num * state.baseline + config.target_den - 1) // config.target_den
        return {
            "phase": state.phase,
            "tick": state.tick,
            "growth_ticks": growth_ticks,
            "recovery_ticks": recovery_ticks,
            "growth_remaining_ticks": config.grow_ticks - growth_ticks,
            "recovery_remaining_ticks": config.recovery_ticks - recovery_ticks,
            "current": state.current,
            "baseline": state.baseline,
            "removed": None if plan is None else plan.removed,
            "remaining_after_cut": state.remaining_after_cut,
            "target": target,
            "hold": state.hold,
            "required_hold": config.hold_ticks,
            "config": {"grow_ticks": config.grow_ticks, "recovery_ticks": config.recovery_ticks,
                       "target_num": config.target_num, "target_den": config.target_den,
                       "hold_ticks": config.hold_ticks},
            "plan": None if plan is None else {
                "x": plan.x, "y": plan.y, "radius": plan.radius,
                "baseline": plan.baseline, "removed": plan.removed, "remaining": plan.remaining},
            "valid": state.phase != "invalid",
            "reason": state.reason,
            "finished": self.finished,
            "can_step": self.can_step,
            "metric": "living cells",
        }
