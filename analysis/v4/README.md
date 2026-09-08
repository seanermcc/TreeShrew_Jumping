# v4 — jump set with two manual onset corrections

v4 is [v3](../v3/README.md) after a second frame-by-frame review pass. **Jump IDs
are unchanged** (still `v3_01`…`v3_17`, with `v3_05` and `v3_07` absent), so every
existing v3 figure and CSV still lines up.

Rebuild:

```bash
E:\TreeShrew_Jumping\analysis\dlc\py310\python.exe v4_build.py
E:\TreeShrew_Jumping\analysis\dlc\py310\python.exe v4_overlay.py
E:\TreeShrew_Jumping\analysis\dlc\py310\python.exe v4_onset_frames.py 2
E:\TreeShrew_Jumping\analysis\dlc\py310\python.exe v4_onset_frames.py 3
E:\TreeShrew_Jumping\analysis\dlc\py310\python.exe v4_figures.py
```

## What changed from v3

| change | jump | detail |
|---|---|---|
| **dropped** | `v3_05` | not a jump — "crawls across" the platform |
| **dropped** | `v3_07` | not a jump — "crawls across" the platform |
| **re-timed** | `v3_06` | onset `f48190 → f48158` (**−0.32 s**). The up→down drop off the platform. The v3 peak-speed detector rejected the true speed max (head keypoints drop out in the motion blur of the drop) and settled on a smaller, cleaner local peak 0.32 s later, by which point the animal was already on the floor. New onset = **onset of the head movement**, per the user's call. |
| **re-timed** | `v3_12` | onset `f68155 → f68255` (**+1.00 s**, = 2.0 s into the 4×-slow v3 review clip). The v3 onset sat on an earlier crawl **plus a tracking artefact**: at f68153 two of the four head keypoints drop out, the centroid jumps 121 px, and the differentiator reports a spurious 5135 px/s. The real head-movement burst is 1 s later (Cam2 ≈ 1000 px/s, Cam3 ≈ 2200). |

**Result: 15 jumps** — 10 `up->up`, 4 `up->down`, 1 `down->up`.
Ballistic labels unchanged: 8 `y`, 6 `n`, 1 blank (`v3_16`). (Both dropped clips,
`v3_05` and `v3_07`, were `up->up` / ballistic `n` crawls, so only those two
buckets shrink; v3 was 12 / 4 / 1 and 8 `y` / 8 `n` / 1 blank.)

The two re-timed onsets are **hand-placed, not peak-detected** — they are flagged
in red in the figures and in the `onset_source` column of `v4_jumps.csv`. Their
`peak_head_speed_pxs` is the cleaned head speed *at the new onset frame*, not a
window maximum.

## Files

| path | what |
|---|---|
| `v4_jumps.csv` | the 15 jumps. New cols vs v3: `onset_source`, `v3_onset_frame`, `d_onset_s` (new − v3 onset, seconds) |
| `_keypoints_v4.csv` | per-frame head keypoints. 13 unchanged jumps = v3's onset-centred re-analysis (`_keypoints_v3.csv`); `v3_06` / `v3_12` re-sliced from the master table (`v3/_keypoints_master.csv`, the original overlapping DLC windows) because their analysis window moved outside the v3 re-cut clip. Master vs v3 re-cut agree to ~0.12 px median (v3 §2). |
| `clips/<id>_<class>.mp4` | Cam2 \| Cam3 review clips, DLC head track drawn on, −1 s…+3 s around onset, 4× slow, red border on the 0–0.2 s jump epoch, head-speed strip with a playhead. Same format as `v3/clips/`. |
| `figures/onset_triplets/fig_onset_triplets_cam{2,3}_g{1,2}.png` | *(v4_onset_frames.py)* overview: every jump's Cam2 (and Cam3) frame at −0.10 s / −0.05 s / 0 s (onset), DLC keypoints on, cropped 520 px around the head. 8 jumps per figure, 2 cols × 4 rows; onset frame red-bordered; the 2 re-timed jumps flagged red. |
| `figures/onset_triplets/frames/<id>_cam<c>_{m100,m050,000}.png` | the individual −0.10 / −0.05 / 0 s crops (`…_clean.png` = same without the keypoint overlay) |
| `figures/onset_triplets/frames/index_cam{2,3}.csv` | exact source frame number, time, and crop box for every snapshot |
| `figures/fig06_head_orientation.png` | *(v4_figures.py)* head orientation vs the travel axis, `up->up` (n=10) vs `up->down` (n=4). Same code as v3 fig06 |
| `figures/fig15_head_axis_stability.png` | *(v4_figures.py)* within-trial head-axis jitter, one number per jump per epoch. Same code as v3 fig15 |
| `figures/fig16_fixation_point.png` | *(v4_figures.py)* fixation point / retinal slip — head axis extended to the landing plane. Same code as v3 fig16 |
| `figures/fig16b_fixation_worked_example.png` | *(v4_figures.py)* fig16 projection drawn on real Cam2 frames (worked example `v3_03`, unchanged from v3) |
| `v4_stability.csv` / `v4_fixation_point.csv` | the per-jump numbers behind fig15 / fig16, recomputed on the v4 set |

## Regenerated figures (`v4_figures.py`)

`v4_figures.py` re-runs **fig06** (from `v3_analysis.py`) and **fig15 / fig16 /
fig16b** (from `v3_stability.py`) on the v4 tables, writing into `v4/figures/`.
The plotting code is untouched — the driver just swaps `v3/v3_jumps.csv` +
`build_keypoints_v3()` for `v4/v4_jumps.csv` + `v4/_keypoints_v4.csv`. What moved
vs v3: `up->up` n 12→10; fig15 medians barely shift (approach circ SD 2.7°→2.9°,
`up->up` approach 5–95 range 2.4°→2.3°, bearing SD 1.0°→0.8°; 12/15 vs 13/17
jumps steady through takeoff); fig16 keeps 12 usable projections but `v3_12`
drops out (its re-timed approach window loses nape/snout coverage) and the
re-timed `v3_06` enters; fig06's `up->down` trace now starts on the platform drop.

## Not regenerated

The rest of the v3 kinematic figures (`fig01`–`fig05`, `fig07`–`fig14`, `fig17`,
the `*_cam3` set) and analysis CSVs (`v3_summary.csv`, `v3_epoch_stats.csv`, …)
are **not** rebuilt. To carry the v4 set through the full analysis, extend the
`v4_figures.py` pattern — point `v3_analysis.py` / `v3_extra.py` at
`v4/v4_jumps.csv` + `v4/_keypoints_v4.csv` and drop the `build_keypoints_v3()`
calls.
