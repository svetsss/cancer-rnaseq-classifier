import pytest

from src.statistical_context import wilson_interval


def test_wilson_interval_for_perfect_test_accuracy_is_not_zero_width() -> None:
    interval = wilson_interval(161, 161)

    assert interval["estimate"] == 1.0
    assert interval["lower"] == pytest.approx(0.976696039648177)
    assert interval["upper"] == 1.0


@pytest.mark.parametrize("successes,total", [(-1, 10), (11, 10), (0, 0)])
def test_wilson_interval_rejects_invalid_counts(successes: int, total: int) -> None:
    with pytest.raises(ValueError):
        wilson_interval(successes, total)
