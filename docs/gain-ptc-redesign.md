# Detector-gain measurement redesign (PTC) — design note

**Status:** deferred / not yet implemented. Captured 2026-06-19 from the DAO-01
(`/media/ronin/miss1/sk`) processing work. Current per-night calibrations were
patched post-hoc with `senpai.cli.measure_gain --write-night`; the changes below
make the gain correct *inside* the pipeline so a reprocess produces it directly.

## Problem

The night calibration reports `conditions.gain_e_per_adu_median`, which is the
**median of a per-frame estimate** (`estimate_gain_from_sky`,
`engine/utils/preprocessing.py`). That per-frame estimate is

    gain = sky_ADU / var_ADU            # var from adjacent-column-diff MAD

i.e. one PTC point with the **intercept forced to zero** — it assumes read
noise = 0. The true relation is `var_ADU = sky_ADU/gain + read²`, so at low sky
the read² term inflates the variance and biases the estimate **low**. Taking the
median over frames does not fix it (it's a median of biased values).

### Evidence (DAO-01 night 0529)

| method | gain (e⁻/ADU) | read noise |
|---|---|---|
| per-frame `sky/var`, night median (current) | **0.153** | — (biased low) |
| same, high-sky decile only (RN negligible) | 0.204 | approaching truth |
| aggregate Theil-Sen fit of **single-frame** points (`var=sky/gain`) | 0.196 | 17.7 e⁻ ❌ unphysical |
| **`measure_gain` pair-difference PTC** (`detector_gain.fit_gain`) | **0.222 ± 0.002** | physical |

Per-frame gain rises monotonically with sky (0.13 → 0.20) — the read-noise
signature. Fitting single-frame points helps (0.153 → 0.196) but still
undershoots and yields a bogus intercept (17.7 e⁻): a single frame's spatial
variance carries **fixed-pattern / flat-residual** structure that the fit
absorbs into the intercept. **Frame-pair differencing cancels that fixed
pattern** — which is why `measure_gain` (already in the tree) is correct.

## Fix: record PTC *points* per frame, fit in aggregate

Stop collapsing to a per-frame gain; record the **pair-difference PTC point**
and fit the night-level slope (the existing `detector_gain` machinery).

| File | Change |
|---|---|
| `engine/observability/detector_gain.py` | none — reuse `find_burst_pairs`, `ptc_point`, `fit_gain` |
| `engine/processing/collect.py` | at batch load (raw arrays in hand, before in-place processing), find same-field consecutive pairs via `find_burst_pairs`, compute `ptc_point(a,b)` → `[(level, var_diff)]`, attach to the run |
| `engine/models/senpai.py` | add `ptc_points: list[tuple[float,float]] = []` to `SenpaiRun` + `SenpaiRunResult`; copy in `to_result()` |
| `engine/observability/calibration.py` | collect `ptc_points` across batch results → `fit_gain` → report gain (slope) **and** read noise (intercept); set `gain_method`. Demote per-frame `sky/var` median to a flagged fallback when a night has too few points |
| tests | synthetic-pair test asserting the aggregate recovers a known gain + read noise |

Net: `night_calibration.json` reports the clean ~0.222 gain **and** a real
read-noise number, no separate `measure_gain` pass. `estimate_gain_from_sky`
survives only as the labeled fallback (too few pairs / no level range).

### Pair-composition reality (DAO-01 cadence)

`find_burst_pairs` keys on same-field `f0→f1`. In this cadence those are almost
entirely the **rate** `f0/f1` halves (~92% rate+rate, ~8% rate+sidereal
calsat-transition, **0 sidereal+sidereal**). So the PTC is rate-only for now;
`ptc_point`'s patch-based lower-envelope rejects the streaked stars (that's how
the 0.222 was obtained). **Decision: report all pairs together; rate-only is
fine for now.** Per-mode (sidereal vs rate) split + divergence flag was
considered and deferred — only becomes useful once a sidereal-burst cadence
exists (below).

## Companion: a burr sidereal-gain measurement mode

A PTC needs (1) **pairs of identical exposures** (same field+exptime, back-to-back
→ difference cancels fixed pattern and, under sidereal tracking, the static
stars) and (2) a **wide range of levels across pairs** (the lever arm pins the
slope = 1/gain; intercept = read²).

Two clean ways to sweep level:

- **Exposure-ladder on one sparse sidereal field** — high-galactic-latitude
  (few bright stars), geometric ladder of exposure times, ≥2 identical
  back-to-back frames per step. Sidereal → stars static → cancel perfectly. Max
  level capped by sky accumulation before bright stars saturate (short lever arm
  on a dark night).
- **Twilight-flat pairs (classic flat PTC, widest clean range)** — pairs of
  identical back-to-back exposures while the twilight sky ramps; the ramp gives
  the level sweep, the field is uniform (no stars), back-to-back keeps sky
  ~constant within a pair. Only change vs current twilight sequence: emit
  **pairs at each level** rather than single auto-exposed frames.

Gotchas to bake in: stay linear (cap top level ≲ 50–60% full well); back-to-back
within a pair; include a few **bias / near-zero-exposure pairs** to anchor read
noise (the intercept; the slope/gain doesn't need them); sparse field for the
star-field variant. All of these feed the same `fit_gain`, and would populate
the sidereal side of an eventual rate-vs-sidereal cross-check.
