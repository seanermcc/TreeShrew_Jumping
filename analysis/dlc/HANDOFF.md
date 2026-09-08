# Tree-shrew jump — head tracking: results & analysis handoff

DeepLabCut head tracking is **done**. This doc is the everything-you-need summary
to start the analysis (in a fresh conversation or by hand).

---

## 1. What was produced

A trained DLC model tracks **5 head keypoints** — `snout, eye, ear, head_top, nape`
— through each of the **27 jumps**, from **Cam2** (clean side profile) and **Cam3**
(oblique 3/4 view), at native **100 fps**, full-res **1024×1024 px**.

Model: HRNet-w32, bottom-up, fine-tuned from SuperAnimal-Quadruped on 209
hand-corrected frames (188 train / 21 test).
**Test error: RMSE 6.35 px** (~one eye-width), mAP 85.7.

### Deliverables

| Path | What |
|---|---|
| `head_pose/cam{2,3}_jump_{NN}_headpose.csv` | per-clip head position + angle, one row per frame |
| `head_pose/_all_headpose.csv` | all 54 clips concatenated |
| `head_pose/_coverage_by_jump.csv` | % of frames tracked, per jump per camera |
| `head3d/jump_{NN}_head3d.csv` | calibration-free pseudo-3D head track (Cam2+Cam3) |
| `qc_videos/jump{NN}_cam{2,3}.mp4` | overlay QC video per clip (54 total, 4x slow-mo) |
| `TreeShrewHead-sem-2026-08-27/videos/*_p60_labeled.mp4` | same videos, original long names |
| `TreeShrewHead-sem-2026-08-27/videos/*_filtered.h5` | raw DLC output (median-filtered) if you want to re-derive things |

---

## 2. `*_headpose.csv` columns

| column | meaning | units / convention |
|---|---|---|
| `clip_frame` | 0–599, frame within the 6 s clip | jump detection ≈ frame 300 |
| `abs_frame` | frame index in the original `20260730_Cam{C}.mkv` | 100 fps |
| `t_abs_s` | absolute time in the recording | seconds |
| `t_rel_s` | time relative to jump detection | `(clip_frame-300)/100`; matches `body3d.py` |
| `cam`, `jump_no` | which camera (2/3) and jump (1–27) | |
| `head_x`, `head_y` | likelihood-weighted centroid of the confident head points | image px, y grows **down** |
| `head_y_up` | `1023 - head_y` | image px, larger = physically **higher** |
| `snout_x/y`, `eye_x/y`, `nape_x/y` | those individual keypoints | image px |
| `*_lik` | DLC confidence for that point | 0–1; treat <0.6 as unreliable |
| `head_angle_deg` | orientation of the **nape→snout** vector | degrees, 0 = +x (frame-right), 90 = up (y flipped) |
| `pitch_from_eye_deg` | orientation of the **eye→snout** vector | same convention — **use this as the primary head-angle proxy** |

**Why two angles:** `pitch_from_eye_deg` uses the two best-tracked points (eye,
snout) and is clean wherever both are present. `head_angle_deg` uses `nape`,
the weakest keypoint, so it is noisier and drops out more often — keep it only
as a cross-check / for a longer lever arm when nape is confident.

Missing keypoints are left blank (NaN) — the model reports nothing rather than
guessing on unresolvable motion-blur frames.

---

## 3. `head3d/jump_{NN}_head3d.csv` — read the caveat

Same recipe as the existing `body3d.py`, but for the head and using Cam3:

| column | source | axis |
|---|---|---|
| `X_along` | Cam2 `head_x` | along the tunnel |
| `Z_up` | Cam2 `head_y_up` | height |
| `Y_across` | Cam3 `head_x` | across the tunnel (**approximate**) |

**Caveat:** `body3d.py` uses Cam2 + **Cam4** (roughly orthogonal). This head
track uses Cam2 + **Cam3** because Cam3 is the only second view that resolves the
eye/snout — but Cam3 is an *oblique* view, so its x-axis is a blend of
along-tunnel and across-tunnel. `X_along` and `Z_up` (both from Cam2) are solid;
`Y_across` is a rough lateral proxy, not a calibrated orthogonal axis. There is
no camera calibration for this rig, so nothing here is metric — pixels only,
relative magnitudes meaningful, absolute distances not.

**Bottom line:** trust the **Cam2 sagittal plane** (`X_along`, `Z_up`, head pitch).
Use `Y_across` only for qualitative "did the head swing left/right".

---

## 4. Coverage — which jumps are usable

`head%` = fraction of the 6 s window where the head centroid was tracked.
`pitch%` = fraction where both snout **and** eye were confident (→ `pitch_from_eye_deg` available).

```
jump   c2_head  c3_head   c2_pitch  c3_pitch
 1       100      54         98       25
 2        97      25         70       18
 3        87      88         76       72
 4        86     100         62       89
 5        74      89         50       56
 6        76      62         51       29
 7        84      77         78       65
 8        66      74         82       70
 9        79      43         42       35
10       100      61        100       79
11       100      92         92       68
12       100      94        100       51
13        63      52         52       43
14        94      37         76       30
15        96      62         92       55
16       100     100         97       51
17        83      40         36       41
18        74      87         56       77
19        86     100         76       65
20        86      96         77       43
21        85      75         72       38
22        96      64         79       26
23        96      92         90       77
24        85      80         96       64
25        52      21         33       12
26       100      50         98       21
27        91     100         85       80
```

- **Cam2: strong** — median 89 % head, usable for the sagittal head kinematics on
  ~24 / 27 jumps. Weak: **8, 13, 25** (fast/edge cases).
- **Cam3: patchy** — median 74 %, and much lower in the ballistic window. This is
  mostly *genuine data limits* (tighter FOV → worse blur, animal at frame edge,
  corner shadow), not model error — verified by eye.
- **Both cams solid → best pseudo-3D:** jumps **4, 11, 12, 16, 19, 23, 24, 27**.
- **Jump 25** is poor on both cameras — inspect the raw clip; may be an odd event.

---

## 5. Recommended analysis approach

Primary signal is **Cam2, sagittal plane**. Suggested pipeline:

1. **Load & clean per jump (Cam2):** take `head_pose/cam2_jump_NN_headpose.csv`.
   Drop rows where the relevant `*_lik < 0.6`. Fill short gaps (≤5 frames) by
   linear interpolation on `t_rel_s`; leave long gaps as NaN. A light
   `scipy.signal.medfilt` (k=5) or Savitzky–Golay on `head_x`, `head_y_up`,
   `pitch_from_eye_deg` removes single-frame jitter — same denoising philosophy as
   `body3d.py` (`DENOISE_KERNEL = 9`).

2. **Kinematics** (all in px and px/s, ×100 fps; **not metric**):
   - head height trajectory: `head_y_up(t)` → apex, launch/land timing, net rise
   - head horizontal: `head_x(t)` → approach speed, in-air horizontal velocity
   - head speed: `hypot(d head_x, d head_y) * 100`
   - **head pitch: `pitch_from_eye_deg(t)`** → pitch at takeoff, pitch sweep
     during flight, pitch at landing. This is the headline new measurement vs the
     old body-centroid pipeline.

3. **Align to an event, not just frame 300.** Frame 300 is the *detection* frame
   from the motion-energy pipeline, not a mechanically defined instant. Define
   takeoff (e.g. first frame head speed exceeds a threshold, or last ground
   contact) and landing per jump, then normalise time to that. `analyze_jumps.py`
   / `body3d.py` already have jump windows you can borrow.

4. **Cross-check against the existing body pipeline.** `body3d/jump_NN_traj.csv`
   has the body-centroid 3D. Head-vs-body: does the head lead the body in pitch?
   Is head height ≈ body height + constant, or does the animal pitch its head
   independently? That contrast is probably the interesting result.

5. **Pool across jumps** using only jumps that clear a coverage bar (say
   `c2_head ≥ 80` and `c2_pitch ≥ 60`): jumps 1, 3, 4?, 7, 10, 11, 12, 14, 15,
   16, 19, 20, 21, 22, 23, 24, 26, 27 — ~18 clean jumps. Report per-jump traces +
   a mean±band aligned to takeoff.

6. **Only if you need lateral head motion / true 3D:** the current `head3d` Y is
   weak. Options: (a) accept qualitative left/right only; (b) redo head tracking
   on **Cam4** as the orthogonal partner (Cam4 sees the head poorly, so expect
   low yield); (c) get a calibration target filmed on the rig and do real DLT/
   triangulation — the proper fix, out of scope here.

### Things to watch
- `head_angle_deg` (nape-based): noisy — prefer `pitch_from_eye_deg`.
- Angles wrap at ±180°; `np.unwrap(np.deg2rad(...))` before differentiating.
- The animal faces **left in Cam2** on some jumps and **right on others** — sign
  of `head_x` velocity and of the pitch angle flips with travel direction.
  Normalise by travel direction before pooling.
- Cam3 `head_x` is not the same axis as Cam2 — never combine them into a single
  scalar without remembering the oblique-view caveat.

---

## 6. Re-running / extending

Everything runs through `E:\TreeShrew_Jumping\analysis\dlc\py310\python.exe`
(self-contained; the `.bat` files set it up). Key scripts:

| script / bat | does |
|---|---|
| `analyze.bat` | re-analyze all 54 clips + re-export head-pose CSVs (skips done) |
| `qc_videos.bat` | render labeled videos for the jumps in its `JUMPS=` line |
| `label_videos.bat` | render labeled videos for **all** 54 clips |
| `07_head_pose.py` | just re-build the CSVs from existing predictions (edit `PCUTOFF` here to change the 0.6 confidence gate) |
| `correct.bat` | re-open the labeler (to add training frames) |

**Targeted refinement round** (if Cam3 or specific jumps need to be better):
```
py310\python.exe -c "import deeplabcut; deeplabcut.extract_outlier_frames(open('PROJECT_CONFIG_PATH.txt').read().strip(), ['TreeShrewHead-sem-2026-08-27/videos/cam3_jump_02.mp4', ...], outlieralgorithm='jump', automatic=True)"
```
then `correct.bat` to label the new frames, then `train.bat`, then `analyze.bat`.
Expected gain on Cam3 is modest (~10–20 pts) because many missing frames are
genuinely unresolvable blur / out-of-frame.

Model snapshot: `TreeShrewHead-sem-2026-08-27/dlc-models-pytorch/iteration-0/TreeShrewHeadAug27-trainset90shuffle1/train/snapshot-best-150.pt`
Eval numbers: `evaluation-results-pytorch/iteration-0/CombinedEvaluation-results.csv`
