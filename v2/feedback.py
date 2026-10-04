"""Presentation-only feedback, independent of the simulation and its RNG.

These helpers consume snapshots. They never change organism states, ownership,
simulation timing, or the supplied arrays/dictionaries.
"""

from collections import deque
import math

import numpy as np


class FrameBlend:
    """Ease RGB presentation in wall time, including draws while paused.

    ``key`` identifies the current match, mode, view, grid, and color scheme.
    A different key or image shape snaps to the new target rather than fading
    unrelated maps together. Only ``rendered`` is interpolated; target pixels
    and simulation state remain exact. Returned uint8 arrays are independent.
    """

    def __init__(self, tau=0.10, max_elapsed=0.10):
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("tau must be finite and positive")
        if not math.isfinite(max_elapsed) or max_elapsed <= 0:
            raise ValueError("max_elapsed must be finite and positive")
        self.tau = float(tau)
        self.max_elapsed = float(max_elapsed)
        self.rendered = None
        self.key = None

    @staticmethod
    def _validate(target):
        if not isinstance(target, np.ndarray) or target.dtype != np.uint8:
            raise TypeError("target must be a uint8 NumPy array")
        if target.ndim != 3 or target.shape[-1] != 3 or min(target.shape[:2]) < 1:
            raise ValueError("target must have shape [height, width, 3]")

    def reset(self, target=None, key=None):
        """Clear the blend, or immediately snap to a supplied RGB target."""
        if target is not None:
            self._validate(target)
        self.key = key
        self.rendered = None if target is None else target.astype(np.float32)
        return None if target is None else target.copy()

    def update(self, target, elapsed_seconds, key=None):
        self._validate(target)
        if self.rendered is None or self.rendered.shape != target.shape or key != self.key:
            return self.reset(target, key)
        elapsed = float(elapsed_seconds)
        # A debugger stop/window drag should not turn into a giant animation
        # jump; invalid or negative timing should not poison the pixel buffer.
        elapsed = min(max(elapsed, 0.0), self.max_elapsed) if math.isfinite(elapsed) else 0.0
        amount = -math.expm1(-elapsed / self.tau)
        self.rendered += (target.astype(np.float32) - self.rendered) * amount
        return np.rint(self.rendered).clip(0, 255).astype(np.uint8)


class ScoreTracker:
    """Exact current scores and a simulation-tick-based observation window.

    Hard mode scores ownership; soft mode scores visible living cells, which
    can overlap. Percentages always use the full board as denominator, so soft
    left/right percentages are not exclusive shares. ``neutral`` means
    unclaimed in hard mode and not visibly alive for either side in soft mode.

    ``span`` is the *actual* observed tick span, not a fabricated 30-tick
    measurement. At 8 ticks per draw it normally spans 32 ticks. We retain the
    sample at or just before the cutoff, without estimating unseen tick counts.
    Call ``reset()`` or change ``epoch`` after reset/clear/culture selection.
    """

    def __init__(self, window_ticks=30):
        if isinstance(window_ticks, bool) or int(window_ticks) != window_ticks or window_ticks < 1:
            raise ValueError("window_ticks must be a positive integer")
        self.window_ticks = int(window_ticks)
        self.reset()

    def reset(self):
        self._samples = deque()
        self._identity = None

    def update(self, stats, epoch=None):
        """Return a fresh summary without modifying ``stats``.

        Repeated observations at one tick replace that tick's sample instead
        of appending/aging history. This also shows paused planting or damage
        immediately. Counts are never eased, even while the board is easing.
        """
        mode = stats["mode"]
        if mode not in ("hard", "soft"):
            raise ValueError("mode must be hard or soft")
        size, step = int(stats["grid_size"]), int(stats["step"])
        if size < 1 or step < 0:
            raise ValueError("grid_size must be positive and step nonnegative")
        total = size * size
        alive_left, alive_right = int(stats["left_alive"]), int(stats["right_alive"])
        contested = int(stats["contested"])
        if mode == "hard":
            if stats["left_territory"] is None or stats["right_territory"] is None:
                raise ValueError("hard mode requires ownership counts")
            left, right = int(stats["left_territory"]), int(stats["right_territory"])
            metric = "owned cells"
            neutral = max(0, total - left - right)
        else:
            left, right = alive_left, alive_right
            metric = "living cells"
            neutral = max(0, total - left - right + contested)

        identity = epoch, mode, size
        if identity != self._identity or (self._samples and step < self._samples[-1][0]):
            self.reset()
            self._identity = identity
        sample = step, left, right
        if self._samples and step == self._samples[-1][0]:
            self._samples[-1] = sample
        else:
            self._samples.append(sample)
        cutoff = step - self.window_ticks
        while len(self._samples) > 1 and self._samples[1][0] <= cutoff:
            self._samples.popleft()

        then, past_left, past_right = self._samples[0]
        span = step - then
        delta_left, delta_right = left - past_left, right - past_right
        lead = left - right
        leader = "left" if lead > 0 else "right" if lead < 0 else None
        if leader is None:
            headline = "LEVEL" if left else "NO SCORE YET"
            reason = f"Both have {left:,} {metric}."
        else:
            headline = f"{leader.upper()} LEADS BY {abs(lead):,}"
            reason = f"{leader.title()} has {abs(lead):,} more {metric}."
        trend = (f"L {delta_left:+,} / R {delta_right:+,} over {span} ticks"
                 if span else "Trend starts after the next simulation tick.")

        # Counts reveal what changed, but cannot distinguish self-regeneration,
        # alpha fluctuations, manual edits, or combat. Never attribute a cause
        # to lost cells or declare a winner/elimination from these observations.
        if not stats["finite"]:
            status = "Non-finite state detected; counts may be unreliable."
        elif not alive_left and not alive_right:
            status = "No visible living cells. Plant a seed to start growth."
        elif mode == "soft":
            status = "Independent growth; living cells may overlap."
        elif not left and not right:
            status = "Living tissue is present; no territory claimed yet."
        else:
            status = f"{neutral:,} unclaimed cells; ownership sets the score."

        return {
            "left": left, "right": right, "metric": metric,
            "lead": lead, "leader": leader, "neutral": neutral,
            "percent": {"left": left * 100.0 / total, "right": right * 100.0 / total,
                        "neutral": neutral * 100.0 / total},
            "delta": {"left": delta_left, "right": delta_right,
                      "lead": delta_left - delta_right},
            "span": span, "headline": headline, "reason": reason,
            "trend": trend, "status": status,
        }
