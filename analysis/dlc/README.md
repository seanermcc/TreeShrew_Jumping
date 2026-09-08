# DeepLabCut head tracking — tree shrew jumps

Goal: track **snout, eye, ear, head_top, nape** to get head position + head
angle during each jump. Runs only on the 27 jump windows (6 s each), not the
full 17 min recording.

## Cameras
- **Cam2** — clean side profile (best: eye + snout crisp)
- **Cam3** — opposite oblique / more frontal (other eye visible)

## Environment
- **`py310/`** — self-contained CPython 3.10.21 with PyTorch CUDA 12.4 +
  `deeplabcut[gui,modelzoo]` in `py310/Lib/site-packages/`. Run everything through
  `py310\python.exe` (the `.bat` files do this). GPU: RTX 3060 Ti 8 GB.
- `venv/` — DEAD (uv-created; its interpreter lived on a path only Claude's
  sandbox could see). Ignore it; can be deleted to reclaim ~6 GB.
- `.bat` files point `HF_HOME`/`TORCH_HOME`/`DLCLIBRARY` at `cache/` so any new
  model downloads land on E:.

## Pipeline
| # | Script | State | Notes |
|---|---|---|---|
| - | `cut_dlc_clips.py 2,3` | done | 54 clips -> `clips/` + `clips/manifest.csv` |
| 1 | `01_create_project.py` | done | project `TreeShrewHead-sem-2026-08-27`, 5 bodyparts |
| 2 | `02_superanimal.py` | done (16 train clips + jump 23) | zero-shot preds + `_labeled_before_adapt.mp4` in `superanimal/` |
| 3 | `03_setup_finetune.py` | done | conversion table -> config; picks 16 training clips |
| 4 | `04c_autolabel.py` | done | 230 frames in `labeled-data/`, ~79% auto-labeled (smoothed SuperAnimal); check sheets in `autolabel_check/` |
| - | `correct.bat` (`correct_labels.py`) | done | user hand-corrected all 16 clips (661 edits). (`label.bat`/napari crashes on this box's GL driver -- unused.) |
| 5a | `05a_clean_empty_frames.py` | done | dropped 21 fully-empty frames -> 209 labeled (188 train / 21 test, 90/10) |
| 5 | `train.bat` (`05_train.py`) | done | HRNet-w32 bottom-up, 150 ep, SuperAnimal backbone init. **test RMSE 6.35 px, mAP 85.7** |
| 6 | `analyze.bat` (`06_analyze.py` + `07_head_pose.py`) | done | all 54 analyzed; CSVs in `head_pose/` + `head3d/` |
| 6b | `label_videos.bat` / `qc_videos.bat` | done | labeled QC videos for all 54 -> `videos/*_p60_labeled.mp4`, copies in `qc_videos/` |

**-> Analysis handoff: see [HANDOFF.md](HANDOFF.md)** — data schema, per-jump coverage, recommended approach.
| 5 | `05_train.py` | pending labels | SuperAnimal-init training dataset + train + evaluate |
| 6 | `06_analyze.py` | pending | all 54 clips -> predictions, filtered, labeled videos |
| 7 | `07_head_pose.py` | pending | head position + head angle CSVs (abs time) + pseudo-3D head track |

Re-run `04 --seed-only` to rebuild seeds without re-extracting frames.

## Coordinate mapping
`clips/manifest.csv` gives `true_start_frame` per clip so DLC (x,y,frame) maps
back to absolute source-video frame / time. Clips are native 100 fps, full
1024x1024 (same scale as `body3d.py` full-res pixels).
