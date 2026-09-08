"""
Re-run DeepLabCut on windows centred on the JUMP ONSET.

Why: the original 27 DLC clips are +-3 s around the *motion-energy detection*
frame. The onset of the real jump often falls near the end of that window, so
the -1 s .. +3 s analysis window is only ~20-80 % covered for several jumps.
Re-cutting around the onset gives 100 % coverage of the window every analysis
downstream actually uses.

Cuts onset-1.5 s .. onset+3.5 s (500 frames; the extra 0.5 s each side is
filter padding, discarded in analysis) for Cam2 and Cam3, then runs
analyze_videos + filterpredictions with the same trained model.

  py310\\python.exe v3_dlc.py cut       # just cut the clips
  py310\\python.exe v3_dlc.py analyze   # just run DLC
  py310\\python.exe v3_dlc.py           # both
"""
import os
import sys
import csv
import cv2
import pandas as pd

import v3_lib as L

PAD_PRE_S, PAD_POST_S = 1.5, 3.5
N_FRAMES = int((PAD_PRE_S + PAD_POST_S) * L.FPS)      # 500
CLIPDIR = os.path.join(L.DLC, "TreeShrewHead-sem-2026-08-27", "videos_v3")
MANIFEST = os.path.join(L.V3, "v3_clip_manifest.csv")


def cut_all():
    os.makedirs(CLIPDIR, exist_ok=True)
    jumps = pd.read_csv(os.path.join(L.V3, "v3_jumps.csv"))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")     # matches the training clips
    rows = []
    for r in jumps.itertuples():
        start = int(r.onset_frame - PAD_PRE_S * L.FPS)
        for cam in (2, 3):
            cap = cv2.VideoCapture(L.SRC.format(cam))
            if not cap.isOpened():
                raise RuntimeError(f"cannot open cam{cam}")
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            ok, frame = cap.read()
            true_start = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            out = os.path.join(CLIPDIR, f"{r.jump}_cam{cam}.mp4")
            vw = cv2.VideoWriter(out, fourcc, L.FPS, (w, h))
            n = 0
            while n < N_FRAMES and ok:
                vw.write(frame)
                n += 1
                ok, frame = cap.read()
            vw.release(); cap.release()
            rows.append(dict(clip=os.path.basename(out), jump=r.jump, cam=cam,
                             onset_frame=r.onset_frame, req_start_frame=start,
                             true_start_frame=true_start, n_frames=n,
                             width=w, height=h, fps=L.FPS))
            flag = "" if (true_start == start and n == N_FRAMES) else "  <-- CHECK"
            print(f"  {os.path.basename(out)}: {n} frames, "
                  f"src[{true_start}..{true_start+n-1}]{flag}", flush=True)
    os.makedirs(L.V3, exist_ok=True)
    with open(MANIFEST, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    bad = [r for r in rows if r["true_start_frame"] != r["req_start_frame"]
           or r["n_frames"] != N_FRAMES]
    print(f"\nwrote {len(rows)} clips + {MANIFEST}")
    print("frame-exact:" , "ALL OK" if not bad else f"{len(bad)} clips off -- {bad}")


def analyze(batch=8):
    """analyze_videos skips clips that already have predictions, so this is
    safe to re-run after a crash -- it picks up where it left off."""
    import glob
    import deeplabcut
    cfg = open(os.path.join(L.DLC, "PROJECT_CONFIG_PATH.txt")).read().strip()
    vids = sorted(glob.glob(os.path.join(CLIPDIR, "v3_*_cam*.mp4")))
    done = len(glob.glob(os.path.join(CLIPDIR, "*_filtered.h5")))
    print(f"analyzing {len(vids)} clips (batch={batch}); {done} already filtered")
    deeplabcut.analyze_videos(cfg, vids, shuffle=1, device="cuda",
                              batch_size=batch, save_as_csv=False)
    deeplabcut.filterpredictions(cfg, vids, shuffle=1, filtertype="median",
                                 windowlength=5, save_as_csv=False)
    print("done")


def n_filtered():
    import glob
    return len(glob.glob(os.path.join(CLIPDIR, "*_filtered.h5")))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "both"
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    if what in ("cut", "both"):
        cut_all()
    if what in ("analyze", "both"):
        analyze(batch)
    if what == "count":
        print(n_filtered())
