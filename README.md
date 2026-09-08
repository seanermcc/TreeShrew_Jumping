# Tree Shrew Jumping

Analysis of high-speed, multi-camera recordings of tree shrew jumping behavior. The project includes motion-based jump detection, animal and head tracking, DeepLabCut model training and inference, 2D/pseudo-3D kinematics, onset review, and publication-oriented figures.

## Project layout

- `analysis/` — Python analysis and visualization pipeline
- `analysis/dlc/` — DeepLabCut head-tracking workflow, labels, predictions, and handoff documentation
- `analysis/body3d/` — body trajectory tables and overview figures
- `analysis/v3/` — reviewed jump set, derived measurements, and figures
- `analysis/v4/` — final 15-jump set with manual onset corrections and regenerated figures
- `JumpingTest_v6_07302026/` — Basler camera configuration files; raw recordings remain local

Start with [`analysis/v4/README.md`](analysis/v4/README.md) for the current jump set and [`analysis/dlc/HANDOFF.md`](analysis/dlc/HANDOFF.md) for the head-tracking results and data schema.

## Data and large files

The repository intentionally excludes machine-specific Python environments, downloaded model caches, raw `.mkv` recordings, generated `.mp4` review clips, multi-gigabyte proxy arrays, and trained `.pt` snapshots. These files remain in the local project directory and can be regenerated or restored separately.

## Environment

The analysis uses Python with NumPy, pandas, SciPy, Matplotlib, OpenCV, PyTorch, and DeepLabCut. See `analysis/dlc/install.ps1` and `analysis/dlc/README.md` for the DeepLabCut environment and workflow used for this project.

## Current result

The v4 dataset contains 15 reviewed jumps: 10 `up->up`, 4 `up->down`, and 1 `down->up`. Two onsets were manually corrected after frame-by-frame review; details and exact outputs are documented in `analysis/v4/README.md`.

