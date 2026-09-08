# Labeling guide — TreeShrewHead

**230 frames** across 16 clips (jumps 1, 4, 8, 12, 16, 19, 23, 27 × Cam2 & Cam3),
**auto-labeled** (≈79% of points already placed by a smoothed SuperAnimal pass).
Your job: correct what's wrong, add what's missing. Budget ~45-75 min.

Preview every clip's auto-labels first: `E:\TreeShrew_Jumping\analysis\dlc\autolabel_check\<clip>.jpg`
(colors: red=snout, green=eye, blue=ear, yellow=head_top, magenta=nape).

## What the auto-labels get right / wrong
- **eye** — almost always correct, tiny nudges at most.
- **ear / head_top / nape** — usually close; nudge onto the exact spot.
- **snout** — good on Cam2. On tight Cam3 close-ups the snout is often off-frame
  and the auto dot sits in a corner — **delete those** (select the snout point,
  press Delete) so it's marked absent.
- **Cam2 mid-flight frames** (~4-5 per clip, heavy blur) — the whole head cluster
  is sometimes on the mid-body/rump. Drag all 5 points onto the real head
  (find the pointed snout + whiskers).
- A point parked exactly at frame center = "no estimate" — place it.

## Launch
napari won't run on this machine (OpenGL driver crash). Use the built-in
corrector instead: in `E:\TreeShrew_Jumping\analysis\dlc`, **double-click
`correct.bat`**.

A window opens showing the first frame with its 5 colored dots. Controls:

| key / action | does |
|---|---|
| drag a dot | move that point |
| `1` `2` `3` `4` `5` | arm snout / eye / ear / head_top / nape, then click to place a missing one |
| `d` or `Delete` | remove the point nearest the cursor (marks it absent) |
| `n` / `p`  or  `->` / `<-` | next / previous frame (auto-saves) |
| `N` / `P` | next / previous clip |
| scroll wheel | zoom under cursor;  `f` fit to animal;  `0` reset zoom |
| `s` | save now |
| `q` or close window | save and quit |

It writes the same `labeled-data/<clip>/CollectedData_sem.h5` that training needs.
Work through all 16 clips (`N` cycles forward). Progress shows in the title bar.

## The 5 keypoints
| point | put it on | notes |
|---|---|---|
| **snout** | tip of the nose | the pointy end, not the mouth |
| **eye** | center of the visible eye | the eye facing the camera |
| **ear** | base of the visible ear where it meets the head | not the ear tip |
| **head_top** | crown of the head, midway between the two ear bases | seeded from ear midpoint — usually needs a nudge up |
| **nape** | back of the neck / base of the skull | where neck meets shoulders |

Only one side is visible in these near-side-on views, so label **one** eye and
**one** ear (whichever faces the camera). If a point is genuinely not visible
(occluded, out of frame), leave it unplaced — do not guess.

## Expect to fix
- **Cam2 mid-flight frames (img~275-330):** SuperAnimal often scrambles the
  head during peak motion blur — seeds may sit on the rump or mid-body. Just
  drag all 5 points to the real head (find the whiskers / pointed snout).
- **head_top / nape:** frequently unseeded — add them.
- **ear:** lowest-confidence seed — check every one.

## When done
Tell me and I'll run:
1. `05_train.py` — create training set (SuperAnimal-initialized) + train (~1-2 h GPU)
2. `06_analyze.py` — all 54 clips → predictions + labeled videos
3. `07_head_pose.py` → head position + head angle CSVs in absolute video time

## Adding more training frames later
If evaluation shows weak spots, `deeplabcut.extract_outlier_frames(CFG, [clip])`
pulls the worst frames from any analyzed clip for a second labeling round.
