# v3 — confirmed jumps: tracked clips + kinematic analysis

Built from your manual pass in `jump_clips_v2/classification_v2.csv`.
Everything here is aligned to **jump onset (t = 0) = the frame of peak Cam2
head-centroid speed** inside the confirmed event, over **−1 s to +3 s**.

Rebuild with:

```bash
E:\TreeShrew_Jumping\analysis\dlc\py310\python.exe v3_build.py && E:\TreeShrew_Jumping\analysis\dlc\py310\python.exe v3_analysis.py && E:\TreeShrew_Jumping\analysis\dlc\py310\python.exe v3_extra.py && E:\TreeShrew_Jumping\analysis\dlc\py310\python.exe v3_stability.py && E:\TreeShrew_Jumping\analysis\dlc\py310\python.exe v3_yaw.py && E:\TreeShrew_Jumping\analysis\dlc\py310\python.exe v3_cam3.py
```

**Cameras:** fig01–fig16 use **Cam2 only** (the sagittal / side view — head
pitch + fore-aft). **fig17** (`v3_yaw.py`) is the yaw check: it brings in Cam3
to test whether the "still head" also holds in the left/right axis Cam2 cannot
see. **`*_cam3.png`** (`v3_cam3.py`) reproduce fig06/09/13/15/16 from Cam3 alone
(oblique, uncalibrated — shapes corroborate, absolute angles do not transfer).

---

## 1. What was kept

| | |
|---|---|
| dropped | the 13 clips you marked `confirmed_jump = no` |
| kept | 16 marked `y` + the 1 `maybe` (v3_01, flagged in `confirmed`) |
| merged | **v2_20+v2_21 → v3_13** and **v2_25+v2_26 → v3_16** (your "jump cut at end" / "combine w/ 25") |
| **total** | **17 jumps** — 12 `up->up`, 4 `up->down`, 1 `down->up` |

Ballistic labels: 8 `y`, 8 `n`, 1 blank (v3_16 — the merged pair; excluded from
ballistic comparisons rather than guessed).

## 2. Tracking was re-run

The original DLC clips were ±3 s around the *motion-energy detection*, but onset
often sits near their end — the −1..+3 s window was only **18–80 %** covered on
several jumps. So 34 new clips were cut centred on the onset and re-analysed
with the same model. Coverage is now **median 86 %, min 56 %**.

Validation: on frames both versions cover, head_x agrees to a **median of
0.12 px** (only 0.05 % of frames differ by >25 px — median-filter edge effects).
Frame indexing is confirmed correct.

## 3. Files

| path | what |
|---|---|
| `clips/v3_NN_<class>.mp4` | Cam2 \| Cam3, **DLC tracking drawn on**, −1..+3 s, 4× slow. Filled dot = keypoint above the 0.6 confidence gate (what analysis uses); hollow = proposed but rejected; white ring = head centroid; white arrow = nape→snout head axis; red border = the 0–0.2 s jump epoch; bottom strip = head-speed trace with a playhead |
| `v3_jumps.csv` | the 17 jumps: onset, window, class, your labels, coverage |
| `v3_summary.csv` | one row per jump — epoch speeds, aiming errors, travel direction |
| `v3_epoch_stats.csv` | mean / SD / **SEM** per epoch, per group (incl. circular stats for angles) |
| `v3_ballistic_physics.csv` | free-flight test numbers |
| `paw_contacts.csv` | **manually annotated** forepaw touchdown point per jump (Cam2 image px, + confidence) — the target behind fig09c |
| `v3_stability.csv` | per-jump within-trial head-axis jitter, epoch by epoch (fig15) |
| `v3_fixation_point.csv` | per-jump fixation-point spread by epoch (fig16) |
| `figures/` | fig01–fig14 |
| `_keypoints_v3.csv` | all 5 keypoints, both cams, per frame (the analysis input) |

Epochs are **pre** `[−1.0, 0)`, **jump** `[0, +0.2)`, **post** `[+0.2, +3.0)`.
Error bars are **SEM across jumps** (SD/√n), n = jumps, not frames.

## 4. Figures

- **fig01** every jump's head speed + mean · **fig02** epoch bars
- **fig03/04** the same split by class
- **fig05** trajectories vs arena geometry (validation)
- **fig06** head orientation vs the travel axis, `up->up` vs `up->down`
- **fig07** head vs body speed, difference, cross-correlation
- **fig08** inter-keypoint distances (out-of-plane head rotation)
- **fig09 / fig09b / fig09c** head aiming vs three targets — where the head
  settles / the platform front edge / the manually-annotated forepaw touchdown
- **fig10** ballistic test · **fig11** angular velocity · **fig12** your ballistic label vs kinematics
- **fig13** representative frames with the keypoints actually used
- **fig14** the forepaw touchdown annotations checked against the real frames
- **fig15** *(v3_stability.py)* **within-trial** head-axis jitter, one number per
  jump per epoch — circular SD of the head angle (raw + linear-detrended), a
  robust peak angular velocity, and the approach 5–95 range of the angle and of
  the head→target bearing. This is the figure that actually backs "the head is
  held still", which fig06 / `v3_epoch_stats.csv` (SDs *across* jumps) do not.
- **fig16** *(v3_stability.py)* fixation point / retinal slip — the nape→snout
  axis extended forward to the landing plane (vertical plane at the platform
  edge for gap crossings, floor plane for drops), tracked frame by frame
- **fig15** bottom row + **fig16b** *(v3_stability.py)* are explainers: fig15's
  bottom row schematics define "head-axis angle" and "circular SD"; fig16b is a
  fig14-style worked example — the fig16 projection drawn on real Cam2 frames
  for one jump (v3_03), with the resulting fixation-height trace beside it
- **fig17** *(v3_yaw.py)* yaw check — Cam2 (pitch) vs Cam3 (pitch + yaw)
  within-approach head-axis jitter, per jump; pseudo-3D azimuth; Cam2 muzzle-
  length foreshortening. Result: for up→up, Cam3 approach SD (1.8°) ≤ Cam2
  (2.4°) and azimuth SD 0.5° → no hidden yaw; the stillness is real, not a
  projection artefact. `v3_yaw.csv`.
- **fig06/09/09b/13/15/16_cam3** *(v3_cam3.py)* Cam3-only reproductions.
  eye→snout axis (Cam3 nape too sparse). fig09b/09c targets (platform edge,
  forepaw touchdown) can't be annotated in the oblique view, so fig09b_cam3
  substitutes the launch point. Cam3 corroborates the Cam2 shape everywhere:
  approach head-axis SD ~4.6°, post ~13°; flat fixation trace in the approach.
  `v3_cam3_stability.csv`, `v3_cam3_fixation.csv`.

## 5. Results

**Speed** (px/s, mean ± SEM across jumps):

| group | pre | jump | post |
|---|---|---|---|
| all (17) | 245 ± 38 | **701 ± 98** | 172 ± 15 |
| up->up (12) | 201 ± 33 | 693 ± 135 | 181 ± 20 |
| up->down (4) | 405 ± 98 | 715 ± 130 | 149 ± 24 |

**Your `ballistic` labels are supported by the tracking** (fig12): `y` jumps reach
966 ± 145 px/s in the jump epoch vs 466 ± 79 for `n` — roughly 2×.

**Head orientation** (nape→snout, mirrored so all jumps travel +x; 0° = travel
direction, −90° = nose down):

| class | pre | jump | post |
|---|---|---|---|
| up->up | −51 ± 8 | −47 ± 4 | **−83 ± 10** |
| up->down | −63 ± 11 | **−39 ± 3** | −46 ± 4 |

They dissociate *after* the event: across-gap jumps end nose-down, down-jumps
level out. n = 4 for `up->down`, so treat that half as provisional.

**Head aiming — three targets** (fig09 / fig09b / fig09c). "Aiming error" = the
angle between the nape→snout head axis and the bearing from the head to the
target (0° = pointing straight at it). Measured over the approach, [−0.6, −0.1] s
before onset, mean ± SD across jumps. Blanked once the head is within 80 px of
the target (bearing undefined there).

| target | up->up | up->down |
|---|---|---|
| a. where the head settles after landing (fig09) | 39 ± 10° | 42 ± 8° |
| b. platform front edge (fig09b) | **15 ± 8°** | ~140° (edge is behind them) |
| c. forepaw touchdown point (fig09c) | 23 ± 9° | **12°** (3 med-conf jumps; 26 ± 29° incl. the 1 low-conf) |

Clean dissociation:
- **Gap crossings** aim the head at the **platform edge they must clear** (15°),
  held rock-steady from −1 s to onset (fig09b density plot). Not at where the
  head ends up (that's up in the air, 39° off).
- **Drops** aim the head **straight at the landing spot on the floor** (~12°),
  and ~140° away from the platform edge behind them. Visual fixation of the
  landing zone.

⚠️ Target (c) is **manually annotated** — the DLC model tracks only the head, so
the forepaw first-contact point was read frame-by-frame off the video
(`paw_contacts.csv`, ~±20 px; a `confidence` column flags the shaky ones). Only
13 of 17 jumps have it: 3 are on-platform repositions that never cross a gap
(v3_01/05/07); v3_14 is a climb whose contact is before the analysis window. The
slow "crawl" jumps have no discrete touchdown, so their annotation is diffuse
(marked low-confidence, dotted in fig09c). fig14 shows every annotation against
its frame.

**Head axis held still through the approach and takeoff** (fig15 / fig16,
`v3_stability.csv`, `v3_fixation_point.csv`). Measured *within* each trial, not
across trials:

| epoch | circ SD of head-axis angle | detrended SD | peak \|angular vel.\| |
|---|---|---|---|
| approach `[−1, −0.15)` | **2.7°** (median) | 1.8° | 149°/s |
| jump `[0, 0.2)` | 4.5° | 2.5° | 352°/s |
| post `[0.2, 3)` | 16.1° | 10.2° | 315°/s |

For `up->up` jumps the head-axis angle spans a **5–95 range of just 2.4°** over
the approach, and the head-axis→platform-edge **bearing** holds to **SD 1.0°**.
13/17 jumps have both the approach and the jump-epoch SD below their post SD —
the head axis is nearly stationary from −1 s through takeoff, then destabilises
~5× during the landing/righting phase. fig16: the point where the head axis
crosses the landing plane stays within ~±10–20 px through the approach on gap
crossings, then sweeps away right at onset. Caveat: the `up->down` jumps (n=4)
are much less steady in the approach (they are re-aiming downward), and the
floor-plane projection for drops is numerically fragile (near-grazing rays;
v3_02 is an outlier).

**Head vs body** (fig07): the body-centroid speed peak **lags** the head, and
head−body speed flips negative after onset (head stabilises while the body is
still moving). ⚠️ t=0 *is defined* as the head-speed peak, so the head peaking
there is circular — the body lag and the sign flip are not.

## 6. ⚠️ Most of these are NOT ballistic leaps

The strongest result here, and it contradicts the framing of "jumps":

- On `up->up` events the head hangs **over the gap for 0.47–1.48 s while dropping
  only 4–54 px**. Nothing unsupported can do that. For v3_15 (1.48 s, 54 px), free
  flight would require a scale of 0.005 px/mm — i.e. the 242 px gap would be
  **48 metres** wide. These are **supported reach-and-bridge** movements.
- Only `up->down` exits (v3_02, v3_06, v3_17) come near a free-fall profile, and
  even those are mostly **controlled climb-downs**: a long gradual lowering, then
  a 2–5 frame drop.

I also tried to **calibrate the rig from physics** — fit gravity to the airborne
head arc, since g = 9806.65 mm/s² would convert px/s² into px/mm and give real
units without a calibration target. **It failed, and the failure is the finding:**
the best parabola fits disagree **3.5×** (3480–12014 px/s²). Gravity is one
number, so those windows are not free flight. **No metric scale is recoverable
this way — a calibration target is still required.**

## 7. Arena geometry (Cam2, measured and validated)

| | value |
|---|---|
| platform surface | y_up = 675 px (feet sit exactly here — verified on frames) |
| left platform front edge | x = 350 px |
| right platform front edge | x = 592 px |
| gap | 242 px |
| floor | y_up ≈ 305 px |

Note the posture difference: the animal lies **prone** on a platform (head only
~20–100 px above it) but stands **upright** on the floor (~185 px). That is why
the landing target is the *measured* head position, not "surface + a constant".

## 8. Caveats

- **Not metric.** Pixels only (see §6). The camera is not telecentric, so depth
  changes apparent scale between jumps.
- **n is small**, especially `up->down` (4) and `down->up` (1). The single
  `down->up` has no SEM and should not be compared statistically.
- **Direction mirroring**: leftward jumps are mirrored (x → −x, angle → 180−angle)
  for all orientation results. Verified: leftward jumps at −137° raw map to −43°,
  matching rightward jumps directly. Speeds are never mirrored.
- **Body track is coarse** — the proxy blob centroid on a 128 px grid upscaled ×8.
  Use it for trends, not precision.
- **Two jumps are weakly tracked**: v3_11 (56 %) and v3_06/v3_07 (72–74 %). They
  drive most of the outliers in the pooled orientation means.
- Speed traces are Savitzky–Golay smoothed (70 ms) for display and epoch means;
  **onset detection uses the unsmoothed signal** so it matches the cut clips.
