from math import sqrt
from typing import TypedDict


class ProportionInterval(TypedDict):
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    successes: int
    total: int
    method: str


def wilson_interval(
    successes: int,
    total: int,
    *,
    z_score: float = 1.959963984540054,
) -> ProportionInterval:
    """Return a two-sided 95% Wilson interval for one binomial proportion."""
    if total <= 0:
        raise ValueError("total must be positive")
    if successes < 0 or successes > total:
        raise ValueError("successes must be between zero and total")

    estimate = successes / total
    denominator = 1 + z_score**2 / total
    center = (estimate + z_score**2 / (2 * total)) / denominator
    margin = z_score * sqrt(estimate * (1 - estimate) / total + z_score**2 / (4 * total**2))
    margin /= denominator
    return {
        "estimate": estimate,
        "lower": max(0.0, center - margin),
        "upper": min(1.0, center + margin),
        "confidence_level": 0.95,
        "successes": successes,
        "total": total,
        "method": "Wilson score interval",
    }
