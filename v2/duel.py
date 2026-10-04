"""Exact, simulation-tick scoring for an opt-in, fixed-length duel.

This module does not advance or inspect a simulation, use its RNG, or import
PyTorch. The caller supplies *post-step* counts for every simulation tick.
With the default rules, ticks 1..60 are warmup and ticks 61..600 are scored.
The initial board at tick 0 is never scored. Rendering, pauses, and wall time
have no effect on the outcome.

Hard mode counts held cells. Soft mode counts living cells, including overlap;
it is an independent-growth comparison, not a territory battle. A result only
describes this round under these rules, not general model quality.
"""

from dataclasses import dataclass
from numbers import Integral


def _integer(value, name, minimum=0):
    """Accept integer counts, but never silently truncate floats or booleans."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class DuelConfig:
    """Round length includes warmup and must leave at least one scoring tick."""

    total_ticks: int = 600
    warmup_ticks: int = 60

    def __post_init__(self):
        total = _integer(self.total_ticks, "total_ticks", 1)
        warmup = _integer(self.warmup_ticks, "warmup_ticks")
        if warmup >= total:
            raise ValueError("warmup_ticks must be less than total_ticks")
        # Normalize other Integral implementations to JSON-safe Python ints.
        object.__setattr__(self, "total_ticks", total)
        object.__setattr__(self, "warmup_ticks", warmup)


@dataclass(frozen=True)
class _RoundState:
    tick: int = 0
    scored_ticks: int = 0
    left_score: int = 0
    right_score: int = 0
    left_current: int | None = None
    right_current: int | None = None
    winner: str | None = None
    invalid_reason: str | None = None


class DuelRound:
    """Accumulate integer cell-ticks, with exact ties and a frozen endpoint.

    ``record_tick`` requires consecutive ticks starting at 1. Duplicate,
    backwards, and skipped ticks raise ValueError without changing the round:
    counts between observations cannot be reconstructed fairly. Call it inside
    the simulation loop, not once per rendered frame. ``clip_steps`` limits a
    requested batch to the remaining horizon; callers must also stop if
    ``finished`` becomes true within the batch (for a non-finite state).

    Invalid arguments raise without mutation. An explicit ``finite=False``
    reports a broken simulation instead: the attempted tick ends the round as
    invalid, its counts are ignored, and no winner is assigned. The caller is
    responsible for checking every simulation tensor relevant to the result.

    After either terminal condition, further recording raises ValueError until
    ``reset``. Snapshots and ``result`` are independent JSON-safe dictionaries;
    changing one cannot alter the internal round or any later export.
    """

    def __init__(self, config=None, mode="hard", board_cells=None):
        if config is None:
            config = DuelConfig()
        if not isinstance(config, DuelConfig):
            raise TypeError("config must be a DuelConfig")
        if mode not in ("hard", "soft"):
            raise ValueError("mode must be hard or soft")
        if board_cells is not None:
            board_cells = _integer(board_cells, "board_cells", 1)
        self._config = config
        self._mode = mode
        self._board_cells = board_cells
        self.reset()

    @property
    def config(self):
        return self._config

    @property
    def mode(self):
        return self._mode

    @property
    def board_cells(self):
        return self._board_cells

    @property
    def tick(self):
        return self._state.tick

    @property
    def finished(self):
        return self._state.invalid_reason is not None or self.tick == self.config.total_ticks

    @property
    def phase(self):
        if self._state.invalid_reason is not None:
            return "invalid"
        if self.finished:
            return "finished"
        return "warmup" if self.tick < self.config.warmup_ticks else "scoring"

    @property
    def result(self):
        """A fresh terminal export, or None while the round is in progress."""
        return self.snapshot() if self.finished else None

    def reset(self):
        """Start another round with the same immutable rules, mode and capacity.

        This only resets scoring. The caller must also reset its simulation and
        RNG if it wants to replay the same seeded match.
        """
        self._state = _RoundState()

    def clip_steps(self, requested):
        """Return the safe batch size; accept zero, reject noninteger requests."""
        requested = _integer(requested, "requested")
        return 0 if self.finished else min(requested, self.config.total_ticks - self.tick)

    def record_tick(self, tick, left, right, *, finite=True):
        """Record one post-step sample and return a new public snapshot.

        A successful tick adds each count once iff ``tick > warmup_ticks``.
        On ``finite=False``, counts may be unavailable and are not inspected.
        Otherwise both counts must be nonnegative integers. When capacity is
        supplied, neither count may exceed it, and hard-mode counts must also
        fit jointly because held territories cannot overlap.
        """
        if self.finished:
            raise ValueError("round is finished; reset before recording another tick")
        tick = _integer(tick, "tick", 1)
        if tick != self.tick + 1:
            raise ValueError(f"expected tick {self.tick + 1}, got {tick}; every tick is required")
        if not isinstance(finite, bool):
            raise ValueError("finite must be a boolean")
        previous = self._state
        if not finite:
            self._state = _RoundState(
                tick=tick, scored_ticks=previous.scored_ticks,
                left_score=previous.left_score, right_score=previous.right_score,
                invalid_reason="Non-finite simulation state; this round has no valid result.")
            return self.snapshot()

        left = _integer(left, "left")
        right = _integer(right, "right")
        if self.board_cells is not None:
            if left > self.board_cells or right > self.board_cells:
                raise ValueError("cell count exceeds board_cells")
            if self.mode == "hard" and left + right > self.board_cells:
                raise ValueError("hard-mode held cell counts cannot overlap")

        scoring = tick > self.config.warmup_ticks
        left_score = previous.left_score + (left if scoring else 0)
        right_score = previous.right_score + (right if scoring else 0)
        winner = None
        if tick == self.config.total_ticks:
            winner = "left" if left_score > right_score else "right" if right_score > left_score else "draw"
        self._state = _RoundState(
            tick=tick, scored_ticks=previous.scored_ticks + int(scoring),
            left_score=left_score, right_score=right_score,
            left_current=left, right_current=right, winner=winner)
        return self.snapshot()

    @staticmethod
    def _average(score, ticks):
        if not ticks:
            return None
        try:
            return score / ticks
        except OverflowError:
            # Python integer scores remain exact even beyond float display
            # range. The numerator and denominator are always in the export.
            return None

    def snapshot(self):
        """Return current progress or the frozen result using builtin types.

        ``scores`` are exact integer cell-ticks. ``averages`` divide these by
        ``scored_ticks`` for display only; they never decide the winner. They
        are None before scoring (or outside the finite float display range).
        ``current`` is unknown before the first recorded tick and after a
        non-finite state. ``final`` exists only for a completed valid round.
        Soft-mode counts and averages may overlap and are not territory shares.
        An invalid result retains its partial, valid scores for diagnostics.
        """
        state = self._state
        current = {"left": state.left_current, "right": state.right_current}
        return {
            "phase": self.phase,
            "tick": state.tick,
            "total_ticks": self.config.total_ticks,
            "warmup_ticks": self.config.warmup_ticks,
            "remaining_ticks": self.config.total_ticks - state.tick,
            "warmup_remaining_ticks": max(0, self.config.warmup_ticks - state.tick),
            "scored_ticks": state.scored_ticks,
            "scores": {"left": state.left_score, "right": state.right_score},
            "averages": {"left": self._average(state.left_score, state.scored_ticks),
                         "right": self._average(state.right_score, state.scored_ticks)},
            "current": current,
            "final": current.copy() if self.phase == "finished" else None,
            "winner": state.winner,
            "finished": self.finished,
            "valid": state.invalid_reason is None,
            "invalid_reason": state.invalid_reason,
            "mode": self.mode,
            "metric": "held cells" if self.mode == "hard" else "living cells",
            "score_unit": "cell-ticks",
        }
