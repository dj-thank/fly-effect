"""Online defect bound for the declared rate model, not biological validation.

For F(x,u)=(1-a)x+a*tanh(Wx+u), ||W||_inf<=g<1 gives
||F(x,u)-F(y,u)||_inf <= q||x-y||_inf, q=1-a+a*g.
If y[t+1]=F(y[t],u[t])+d[t], E[t+1]=q*E[t]+||d[t]||_inf
bounds ||x[t+1]-y[t+1]||_inf for a shared input and initial error E[0].
No intact trajectory is an input to this online monitor. Bounds are an exact-
arithmetic mathematical statement; this implementation is not interval
arithmetic and does not claim a rigorous floating-point enclosure.
"""
from __future__ import annotations
from dataclasses import dataclass
import math


def contraction(gain: float, alpha: float) -> float:
    if (isinstance(gain, bool) or isinstance(alpha, bool)
            or not math.isfinite(gain) or not math.isfinite(alpha)
            or not 0 <= gain < 1 or not 0 < alpha <= 1):
        raise ValueError('Require 0<=gain<1 and 0<alpha<=1')
    return 1 - alpha + alpha * gain


@dataclass
class DefectMonitor:
    """Scalar infinity-norm bound; caller supplies the ACTUAL local defect."""
    q: float
    bound: float = 0.0
    peak: float = 0.0
    steps: int = 0

    def __post_init__(self):
        if (isinstance(self.q, bool) or not math.isfinite(self.q)
                or not 0 <= self.q < 1 or not math.isfinite(self.bound)
                or self.bound < 0):
            raise ValueError('Finite contractive q and nonnegative initial bound required')
        self.peak = self.bound
        self.steps = 0

    def advance(self, defect_inf: float) -> float:
        if (isinstance(defect_inf, bool) or not math.isfinite(defect_inf)
                or defect_inf < 0):
            raise ValueError('Finite nonnegative infinity-norm defect required')
        value = self.q * self.bound + defect_inf
        if not math.isfinite(value):
            raise ValueError('Bound overflow')
        self.bound = value
        self.peak = max(self.peak, value)
        self.steps += 1
        return value
