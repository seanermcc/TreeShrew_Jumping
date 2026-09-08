# v4 — reviewed jump set (3 drops, 1 onset correction)

v4 is [v3](../v3/README.md) after two frame-by-frame review passes. **Jump IDs
are unchanged** (still `v3_01`…`v3_17`, with `v3_05`, `v3_07`, `v3_12` absent), so
every existing v3 figure and CSV still lines up.

Rebuild:

```bash
E:\TreeShrew_Jumping\analysis\dlc\py310\python.exe v4_build.py
E:\TreeShrew_Jumping\analysis\dlc\py310\python.exe v4_overlay.py
E:\TreeShrew_Jumping\analysis\dlc\py310\python.exe v4_onset_frames.py 2
E:\TreeShrew_Jumping\analysis\dlc\py310\python.exe v4_onset_frames.py 3
E:\TreeShrew_Jumping\analysis\dlc\py310\python.exe v4_figures.py
E:\TreeShrew_Jumping\analysis\dlc\py310\python.exe v4_fig09.py
E:\TreeShrew_Jumping\analysis\dlc\py310\python.exe v4_paw_snaps.py
```

## What changed from v3

| change | jump | detail |
|---|---|---|
| **dropped** | `v3_05` | not a jump — "crawls across" the platform |
| **dropped** | `v3_07` | not a jump — "crawls across" the platform |
| **dropped** | `v3_12` | **a fall, not a real jump** ("crawls + falls a bit"). Briefly kept + re-timed in the first v4 pass; the 2026-09-08 review reclassified it and pulled it from every downstream analysis and from `clips/`. |
| **re-timed** | `v3_06` | onset `f48190 → f48158` (**−0.32 s**). The up→down drop off the platform. The v3 peak-speed detector rejected the true speed max (head keypoints drop out in the motion blur of the drop) and settled on a smaller, cleaner local peak 0.32 s later, by which point the animal was already on the floor. New onset = **onset of the head movement**, per the user's call. |

**Result: 14 jumps** — 9 `up->up`, 4 `up->down`, 1 `down->up`.
Ballistic labels: 7 `y`, 6 `n`, 1 blank (`v3_16`). (The three dropped clips were
all `up->up`; `v3_05`/`v3_07` ballistic `n`, `v3_12` ballistic `n`. v3 was
12 / 4 / 1.)

`v3_06`'s onset is **hand-placed, not peak-detected** — flagged in red in the
figures and in the `onset_source` column of `v4_jumps.csv`; its
`peak_head_speed_pxs` is the cleaned head speed *at that frame*, not a window max.

## Files

| path | what |
|---|---|
| `v4_jumps.csv` | the 14 jumps. New cols vs v3: `onset_source`, `v3_onset_frame`, `d_onset_s` (new − v3 onset, seconds) |
| `_keypoints_v4.csv` | per-frame head keypoints. 12 unchanged jumps = v3's onset-centred re-analysis (`_keypoints_v3.csv`); `v3_06` re-sliced from the master table (`v3/_keypoints_master.csv`, the original overlapping DLC windows) because its analysis window moved outside the v3 re-cut clip. Master vs v3 re-cut agree to ~0.12 px median (v3 §2). |
| `clips/<id>_<class>.mp4` | Cam2 \| Cam3 review clips, DLC head track drawn on, −1 s…+3 s around onset, 4× slow, red border on the 0–0.2 s jump epoch, head-speed strip with a playhead. Same format as `v3/clips/`. |
| `figures/onset_triplets/fig_onset_triplets_cam{2,3}_g{1,2}.png` | *(v4_onset_frames.py)* overview: every jump's Cam2 (and Cam3) frame at −0.10 s / −0.05 s / 0 s (onset), DLC keypoints on, cropped 520 px around the head. 14 jumps, 7 per figure, 2 cols × 4 rows; onset frame red-bordered; the re-timed `v3_06` flagged red. |
| `figures/onset_triplets/frames/<id>_cam<c>_{m100,m050,000}.png` | the individual −0.10 / −0.05 / 0 s crops (`…_clean.png` = same without the keypoint overlay) |
| `figures/onset_triplets/frames/index_cam{2,3}.csv` | exact source frame number, time, and crop box for every snapshot |
| `figures/fig06_head_orientation.png` | *(v4_figures.py)* head orientation vs the travel axis, `up->up` (n=9) vs `up->down` (n=4). Same code as v3 fig06 |
| `figures/fig15_head_axis_stability.png` | *(v4_figures.py)* within-trial head-axis jitter, one number per jump per epoch. Same code as v3 fig15 |
| `figures/fig16_fixation_point.png` | *(v4_figures.py)* fixation point / retinal slip — head axis extended to the landing plane. Same code as v3 fig16 |
| `figures/fig16b_fixation_worked_example.png` | *(v4_figures.py)* fig16 projection drawn on real Cam2 frames (worked example auto-picked = best-covered `up->up` crossing) |
| `figures/fig09_target_aiming.png` | *(v4_fig09.py)* head-axis aiming error vs the MEASURED head landing position, `up->up` (n=9) vs `up->down` (n=4). Same code as v3 fig09 |
| `figures/fig09b_edge_aiming.png` | *(v4_fig09.py)* head-axis aiming error vs the landing-platform FRONT EDGE, `up->up` gap crossings only. Same code as v3 fig09b |
| `figures/fig09c_paw_aiming.png` | *(v4_fig09.py)* head-axis aiming error vs the MANUAL forepaw touchdown point (`paw_contacts_v4.csv` if present, else `v3/paw_contacts.csv`), `up->up` (n=9) vs `up->down` (n=4). Same code as v3 fig09c |
| `figures/fig09d_paw_aiming_examples.png` | *(v4_fig09.py)* NEW. fig16b-style worked example of fig09c: 3 `up->up` gap crossings (2 near the group median approach aiming error, 1 outlier), Cam2 frames at t = −0.6…+0.15 s with the nape→snout head axis vs the straight-to-forepaw-target line drawn on, plus the per-jump aiming-error trace. Picks auto-selected (with the re-labelled points: `v3_10` / `v3_15` representative, `v3_03` outlier) or passed as argv |
| `figures/fig09e_paw_contact_frames_{1,2}.png` | *(v4_paw_snaps.py)* check sheet: every jump's landing, Cam2, onset → +0.40 s in 50 ms steps, current touchdown point drawn on. Diagnostic for spotting bad annotations |

### Re-labelling the forepaw touchdown points

`v4_paw_gui.py` (launch: `label_paws.bat`, or `... v4_paw_gui.py v3_03 v3_09` for a
subset) — a matplotlib/Tk labeller (no OpenGL, works on this machine). Left: a
Cam2 contact sheet, onset −0.05 → +0.55 s; click the frame where the forepaw
plants. Right: that frame, zoomed; click/drag to put the target on the paw. Head
axis + head→target line + aiming angle shown live. Keys: `, .` frame, `[ ]` zoom,
`n/p` jump, `c` confidence, `s` surface, `x` "no discrete contact", `q` save+quit.
Autosaves to **`v4/paw_contacts_v4.csv`** (columns of `v3/paw_contacts.csv` plus
`onset_frame`, `dframe`, `status`, `edited`). `v4_fig09.py`, `v4_figures.py` and
`v4_paw_snaps.py` use it automatically when present; rows with
`status` in `{no_contact, dropped, none}` are skipped by the aiming figures
(`v3_analysis.PAW_CONTACTS_PATH` + status filter). The user re-labelled all
touchdown points on 2026-09-08 — the corrected points sit noticeably closer to
the head axis (up->up approach `|aim err|` ~24° → ~19°).
| `v4_stability.csv` / `v4_fixation_point.csv` | the per-jump numbers behind fig15 / fig16, recomputed on the v4 set |

## Regenerated figures (`v4_figures.py`)

`v4_figures.py` re-runs **fig06** (from `v3_analysis.py`) and **fig15 / fig16 /
fig16b** (from `v3_stability.py`) on the v4 tables, writing into `v4/figures/`.
The plotting code is untouched — the driver just swaps `v3/v3_jumps.csv` +
`build_keypoints_v3()` for `v4/v4_jumps.csv` + `v4/_keypoints_v4.csv`. What moved
vs v3: `up->up` n 12→9; fig15 says "all 14 jumps", medians approach circ SD
2.7°, jump 4.6°, post 15.1°, up->up approach 5–95 range 2.3°, bearing SD 0.8°,
11/14 jumps steady through takeoff; fig16 = 11 usable projections; fig06's
`up->down` trace starts on the platform drop.

`v4_fig09.py` re-runs **fig09 / fig09b / fig09c** (from `v3_analysis.fig_target`,
code untouched) on the v4 `W` and adds **fig09d**, a fig16b-style worked example.
Counts: `up->up` n=9 (fig09/09b/09c), `up->down` n=4. up->up approach aiming
(re-labelled points): ~16° at the platform edge (fig09b), ~19° at the forepaw
touchdown point (fig09c) — v3_03 the lone outlier at ~37°.

## Not regenerated

The rest of the v3 kinematic figures (`fig01`–`fig05`, `fig07`–`fig08`,
`fig10`–`fig14`, `fig17`, the `*_cam3` set) and analysis CSVs (`v3_summary.csv`,
`v3_epoch_stats.csv`, …) are **not** rebuilt. To carry the v4 set through the full analysis, extend the
`v4_figures.py` pattern — point `v3_analysis.py` / `v3_extra.py` at
`v4/v4_jumps.csv` + `v4/_keypoints_v4.csv` and drop the `build_keypoints_v3()`
calls.
