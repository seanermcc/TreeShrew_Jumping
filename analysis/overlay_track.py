"""
Burn the tracked centroid onto real full-resolution frames, for visual QA.

Answers "what is actually being tracked?" directly: the marker is drawn at
the background-subtracted blob centroid used throughout track_animal.py /
body3d.py -- the center of mass of the whole animal silhouette in that
camera's view, NOT a specific landmark (no eye/nose/head detection is done
anywhere in this pipeline). Where the marker sits on the animal as posture
changes (crouched vs. extended vs. mid-leap) shows exactly what that implies.
"""
import os
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd
from scipy.signal import medfilt

FFMPEG = os.environ.get("FFMPEG_BIN", "ffmpeg")
FPS = 100.0
PROXY_TO_FULL = 8


def interp_nan(a):
    a = a.astype(np.float64).copy()
    bad = ~np.isfinite(a)
    if bad.any() and not bad.all():
        idx = np.arange(len(a))
        a[bad] = np.interp(idx[bad], idx[~bad], a[~bad])
    return a


def grab_frame(path, t):
    cmd = [FFMPEG, "-v", "error", "-nostdin", "-ss", f"{t:.3f}", "-i", path,
           "-frames:v", "1", "-pix_fmt", "gray", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    n = int(len(raw) ** 0.5)
    return np.frombuffer(raw, np.uint8).reshape(n, n)


def main():
    src_dir, proxy_dir, csv_path = sys.argv[1], sys.argv[2], sys.argv[3]
    cam = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    jump_no = int(sys.argv[5])
    out_path = sys.argv[6]

    df = pd.read_csv(csv_path)
    jumps = df[df.is_jump].sort_values("t_s").reset_index(drop=True)
    t_center = float(jumps.iloc[jump_no - 1].t_s)

    tr = np.load(os.path.join(proxy_dir, f"cam{cam}_track.npz"))
    cx, cy = interp_nan(tr["cx"]), interp_nan(tr["cy"])
    f_center = int(round(t_center * FPS))
    lo, hi = f_center - 300, f_center + 300
    cxw, cyw = medfilt(cx[lo:hi], 9), medfilt(cy[lo:hi], 9)

    offsets = np.arange(-3.0, 3.01, 0.5)
    tiles = []
    video_path = os.path.join(src_dir, f"20260730_Cam{cam}.mkv")
    for t_rel in offsets:
        t_abs = t_center + t_rel
        idx = int(round((t_rel + 3.0) * FPS))
        idx = max(0, min(len(cxw) - 1, idx))
        px, py = cxw[idx] * PROXY_TO_FULL, cyw[idx] * PROXY_TO_FULL

        frame = grab_frame(video_path, t_abs)
        img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        cv2.drawMarker(img, (int(px), int(py)), (0, 0, 255),
                        markerType=cv2.MARKER_CROSS, markerSize=40, thickness=3)
        cv2.circle(img, (int(px), int(py)), 22, (0, 0, 255), 3)
        label = f"t={t_rel:+.1f}s" + ("  (JUMP)" if abs(t_rel) < 1e-6 else "")
        cv2.rectangle(img, (0, 0), (260, 40), (0, 0, 0), -1)
        cv2.putText(img, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2)
        tiles.append(cv2.resize(img, (300, 300)))

    rows = [np.hstack(tiles[i:i+7]) for i in range(0, len(tiles), 7)]
    maxw = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.zeros((r.shape[0], maxw - r.shape[1], 3), np.uint8)])
            if r.shape[1] < maxw else r for r in rows]
    sheet = np.vstack(rows)
    cv2.imwrite(out_path, sheet)
    print(f"wrote {out_path}  ({sheet.shape[1]}x{sheet.shape[0]})")


if __name__ == "__main__":
    main()
