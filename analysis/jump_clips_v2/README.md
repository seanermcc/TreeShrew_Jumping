# jump_clips_v2 — cleaned single-jump events

Rebuilt from the 27 raw motion-energy clips in `../jump_clips` using the
DeepLabCut **head track** (`../dlc/head_pose/_all_headpose.csv`) to find the
*actual* relocation events, deduplicate the multiple detections that fire around
one jump, and drop triggers where the animal did not actually go anywhere.

Built by `../classify_jumps_v2.py`
(run: `..\dlc\py310\python.exe ..\classify_jumps_v2.py`).

## Files

| file | what |
|---|---|
| `v2_NN.mp4` | Cam2 (side) \| Cam3 (oblique) side-by-side, **1 s before takeoff → 1 s after landing**, 4× slow-mo. Red border = the takeoff→landing window. Title bar shows which original `jump_clips` detection(s) it came from and the up/down class. |
| `classification.csv` | one row per clip — see columns below |

## How the events were found

1. The 27 DLC windows (each ±3 s around a raw detection) were stitched into
   continuous per-segment head traces (Cam2 primary — clean side profile;
   Cam3 = oblique cross-check).
2. Head centroid de-glitched (Hampel + median), head speed derived.
3. Each segment split into **movement bouts** separated by ≥0.3 s of stillness;
   bouts <1 s apart merged (max 5 s span) → ~one bout per real relocation.
4. A bout is kept as an event if the head actually moved between its
   before/after resting positions (≥120 px net, or ≥85 px vertical, or a very
   fast peak, or it covers an original detection and moved ≥70 px).
5. **Class** from resting head height (`head_y_up`) just before vs just after:
   platform ≈ 700–820 px, floor ≈ 430–540 px (Cam2; Cam3 used to break ties).

Every original detection 01–27 is represented by at least one v2 clip.

## Columns in classification.csv

| column | meaning |
|---|---|
| `clip` | `v2_NN`, matches `v2_NN.mp4` |
| `orig_detections` | which `jump_clips/jump_NN` this came from (`+` = merged; `(none)` = a real bout the raw detector missed) |
| `class_updown` | **`up->up`** platform↔platform / along platform · **`up->down`** platform→floor · **`down->up`** floor→platform |
| `confirmed_jump` | **you fill in** — `y` / `n` (is this actually a jump?) |
| `ballistic` | **you fill in** — `y` = true launched jump · `n` = walk / climb / crawl |
| `t_takeoff_s`, `t_land_s` | event start/end, absolute seconds in `20260730_CamX.mkv` (100 fps) |
| `dur_ms` | landing − takeoff |
| `peak_head_speed_pxs` | peak Cam2 head speed during the event (px/s; not metric). Flagged `[glitch]` in `claude_assessment` if it may be a 1-frame tracking spike |
| `dz_px` | head height change, + = ended higher |
| `net_disp_px` | straight-line head displacement, before→after |
| `cam2_z_pre/post`, `cam3_z_pre/post` | resting head height each side (the classifier inputs) |
| `track_quality` | `ok` / `gappy` (head lost for part of the fast phase) / `glitch` |
| `gap_frac` | fraction of the event where the Cam2 head track was interpolated |
| `clip_start_s`, `clip_end_s` | window of the mp4 |
| `claude_assessment` | first-pass guess, **not authoritative** — your `confirmed_jump` / `ballistic` win |
| `notes` | yours |

## Caveats / where to look closely

- The arena (from frame-by-frame review) is **one elevated clear platform + the
  optical-table floor**. `up->down` (clips 02, 08, 21, 30) and `down->up`
  (clip 22) are unambiguous level changes. The `up->up` clips are fast
  traverses / lunges *along* the platform — whether any of those count as a
  "jump" (vs the animal just running across) is exactly the `ballistic` call.
- A few long slow traverses got split into two adjacent clips
  (e.g. 09+10 = jumps 08/09). Merge them in your notes if you like.
- `peak_head_speed_pxs` on clip 19 (≈5100) is inflated by a 1-frame tracking
  flicker — the move is real but not that fast.
