"""`validation.max_shift_trials` sets how many shift candidates each frame-to-frame search tries.

Both searches (sidereal -> rate and rate -> rate) take cross-correlation peaks in order, validating each,
until one passes or the budget runs out. The budget was a hardcoded 10. On real collects the correct
shift can sit just past it: behind a sign-flipped decoy and side-lobes of peaks already tried. When
that happens the hop fails, and the frames after it lose their WCS. So the budget is configuration.
The default stays 10, and a deployment that wants more raises it.

The synthetic frames below are shared with ``test_prod_stability_guards``:
- the sidereal -> rate pair validates only at its 10th candidate, the real-data failure in miniature;
- every rate -> rate candidate fails, so that search uses its whole budget.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
import test_prod_stability_guards as frames

from senpai.core.config import ValidationConfig, get_config, initialize_config
from senpai.engine.detection.streak import rate_rate, rate_sidereal
from senpai.engine.models.senpai import FrameShift


@pytest.fixture(scope="module", autouse=True)
def _config() -> None:
    initialize_config(frames.CONFIG_DIR / "burr.yaml")
    get_config().plotting.debug = False


@pytest.fixture
def budget(monkeypatch):
    """Set the configured budget for one test."""

    def set_budget(n: int) -> None:
        monkeypatch.setattr(get_config().validation, "max_shift_trials", n, raising=False)

    return set_budget


def _count_calls(monkeypatch, module, name: str) -> list:
    calls: list = []
    original = getattr(module, name)
    monkeypatch.setattr(module, name, lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    return calls


def test_default_is_upstreams_ten():
    assert ValidationConfig().max_shift_trials == 10


def _sidereal_hop(monkeypatch, budget, n):
    budget(n)
    tries = _count_calls(monkeypatch, rate_sidereal, "streak_parameters_from_xcorr")
    np.random.seed(1234)
    sidereal = frames._solved_sidereal_frame(0, datetime(2024, 1, 1, 0, 0, 0), exptime=2.0)
    rate = frames._rate_frame(1, datetime(2024, 1, 1, 0, 0, 2), exptime=2.0, seed=3)
    shift = FrameShift(source_index=0, target_index=1)
    rate_sidereal.solve_rate_from_sidereal(sidereal, rate, shift)
    return len(tries), shift


def test_sidereal_to_rate_gives_up_at_the_configured_budget(monkeypatch, budget):
    """Its shift validates only at the 10th candidate, so a budget of 9 runs out first."""
    tries, shift = _sidereal_hop(monkeypatch, budget, 9)
    assert tries == 9
    assert shift.is_valid is False


def test_a_larger_budget_reaches_a_shift_found_late(monkeypatch, budget):
    tries, shift = _sidereal_hop(monkeypatch, budget, 14)
    assert tries == 10
    assert shift.is_valid is True


@pytest.mark.parametrize("n", [3, 14])
def test_rate_to_rate_tries_the_configured_number(monkeypatch, budget, n):
    budget(n)
    tries = _count_calls(monkeypatch, rate_rate, "validate_proposed_shift")
    np.random.seed(1234)
    a = frames._rate_frame(0, datetime(2024, 1, 1, 0, 0, 0), exptime=2.0, seed=3)
    b = frames._rate_frame(1, datetime(2024, 1, 1, 0, 0, 10), exptime=2.0, seed=4)
    a.starfield = frames._solved_sidereal_frame(0, datetime(2024, 1, 1, 0, 0, 0), exptime=2.0).starfield
    shift = FrameShift(source_index=0, target_index=1)

    rate_rate.solve_rate_from_rate(a, b, shift)

    assert len(tries) == n
    assert shift.is_valid is False
