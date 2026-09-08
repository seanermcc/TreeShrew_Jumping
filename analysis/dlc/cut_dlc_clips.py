"""
Cut per-camera jump-window clips for DeepLabCut.

For every jump in the inventory, extract a 6 s window (3 s before to 3 s after
the detection frame) from each requested camera's source .mkv, at native
resolution and native 100 fps (NO slow-mo, unlike cut_jump_clips.py which
makes 4x montages for human review).

Output: dlc/clips/cam{C}_jump_{NN}.mp4  (one single-view clip per jump/cam)
plus dlc/clips/manifest.csv mapping each clip to its jump number, absolute
source-frame range and absolute time, so DLC coordinates/time can be mapped
back to the original recording.
"""
import os
import sys
import csv

import cv2
import numpy as np
import pandas as pd

FPS = 100.0
HALF_WINDOW_S = 3.0
N_FRAMES = int(2 * HALF_WINDOW_S * FPS)  # 600

SRC_DIR = r"H:\TreeShrew_Jumping\JumpingTest_v6_07302026"
INVENTORY = r"E:\TreeShrew_Jumping\analysis\proxy\jump_inventory_cam2.csv"
OUT_DIR = r"E:\TreeShrew_Jumping\analysis\dlc\clips"


def cut(cam, jump_no, start_frame, writer_fourcc):
    src = os.path.join(SRC_DIR, f"20260730_Cam{cam}.mkv")
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {src}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, frame = cap.read()
    true_start = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    if not ok:
        cap.release()
        raise RuntimeError(f"cam{cam} jump{jump_no}: read failed at {start_frame}")

    out_path = os.path.join(OUT_DIR, f"cam{cam}_jump_{jump_no:02d}.mp4")
    vw = cv2.VideoWriter(out_path, writer_fourcc, FPS, (w, h))
    n = 0
    while n < N_FRAMES and ok:
        vw.write(frame)
        n += 1
        ok, frame = cap.read()
    vw.release()
    cap.release()
    return out_path, true_start, n, w, h


def main():
    cams = [int(c) for c in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["2", "3"])]
    os.makedirs(OUT_DIR, exist_ok=True)

    # mp4v is universally writable by OpenCV and readable by DLC/OpenCV.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    df = pd.read_csv(INVENTORY)
    jumps = df[df.is_jump].sort_values("t_s").reset_index(drop=True)
    print(f"{len(jumps)} jumps x {len(cams)} cams = {len(jumps) * len(cams)} clips")

    rows = []
    for i, row in jumps.iterrows():
        jump_no = i + 1
        t_jump = float(row.t_s)
        det_frame = int(round(t_jump * FPS))
        start_frame = max(0, det_frame - int(HALF_WINDOW_S * FPS))
        for cam in cams:
            out_path, true_start, n, w, h = cut(cam, jump_no, start_frame, fourcc)
            rows.append(dict(
                clip=os.path.basename(out_path), cam=cam, jump_no=jump_no,
                t_jump_s=round(t_jump, 3), det_frame=det_frame,
                req_start_frame=start_frame, true_start_frame=true_start,
                n_frames=n, width=w, height=h, fps=FPS,
            ))
            print(f"  cam{cam} jump_{jump_no:02d}: {n} frames, "
                  f"src[{true_start}..{true_start + n - 1}] -> {os.path.basename(out_path)}",
                  flush=True)

    man = os.path.join(OUT_DIR, "manifest.csv")
    with open(man, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {man}")


if __name__ == "__main__":
    main()
