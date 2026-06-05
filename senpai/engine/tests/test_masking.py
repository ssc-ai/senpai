"""Regression tests for streak masking — the vectorized saturation/border
removers and the bounded flood fill.

These guard the fix for the rate->rate hang: removing saturated/border streaks
one-at-a-time with full-frame ops was O(n_sources x frame_size) and an
unbounded flood fill could run away across a frame connected by a dead
row/column. The removers are now single-pass label/fill, and the flood fill is
bounded by default.
"""

from __future__ import annotations

import numpy as np

from senpai.engine.detection.streak.masking import (
    remove_border_crossing_streaks,
    remove_near_saturation_streaks,
    remove_streak_at_point,
)


def test_remove_near_saturation_removes_only_saturated_blobs():
    img = np.full((60, 60), 100.0)
    img[10:13, 10:13] = 65535.0  # saturated source
    img[40:43, 40:43] = 5000.0   # bright but not near-saturation

    out, n = remove_near_saturation_streaks(img.copy(), "uint16")

    assert n == 1
    # Saturated blob filled down to ~mean, well below the 0.9*65535 cut.
    assert out[11, 11] < 0.9 * 65535
    assert out.max() < 0.9 * 65535
    # The non-saturated bright source is untouched.
    assert out[41, 41] == 5000.0


def test_remove_near_saturation_noop_when_unsaturated():
    img = np.full((40, 40), 100.0)
    img[20:22, 20:22] = 5000.0
    out, n = remove_near_saturation_streaks(img.copy(), "uint16")
    assert n == 0
    assert np.array_equal(out, img)


def test_remove_border_crossing_removes_only_border_seeded_blobs():
    img = np.full((60, 60), 100.0)
    img[0:3, 28:31] = 5000.0    # blob crossing the top border
    img[30:33, 30:33] = 5000.0  # interior blob, must survive

    out = remove_border_crossing_streaks(img.copy())

    assert out[1, 29] < 5000.0       # border blob removed
    assert out[31, 31] == 5000.0     # interior blob kept


def test_flood_fill_is_bounded_on_fully_connected_frame():
    # Every pixel is above threshold (mimics a frame bridged by a dead line):
    # an unbounded flood fill would traverse all 360k pixels; the bounded
    # default must cap it well below that.
    img = np.full((600, 600), 1000.0)
    img[300, 300] = 65535.0
    before = img.copy()

    out = remove_streak_at_point(img.copy(), (300, 300), fill_min=500.0)

    n_changed = int(np.sum(out != before))
    assert n_changed > 0
    assert n_changed <= 200_000          # bounded by max_pixels default
    assert n_changed < before.size       # did not consume the whole frame
