"""
Speed of the tracked body centroid, -3s to +3s around each jump.

Speed here is |d(centroid)/dt| in Cam2's image plane (full-res-equivalent
px/s, i.e. proxy px * 8 * 100 fps). It is NOT metric (no calibration target
exists for this rig) and it reflects the whole-body blob centroid, so it mixes
true translation with posture change (crouch/extend/leap) -- see
overlay_track.py output for what that centroid actually sits on. For "how
vigorously is the animal moving, and when" that conflation is a feature: any
fast whole-body event (translating, launching, landing) shows up as a speed
peak, which is exactly the relative-speed signal requested.

Cam2 only (single best-quality view) -- avoids the cross-camera-axis mixing
used in the pseudo-3D reconstruction, which is a separate, noisier estimate.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import medfilt

FPS = 100.0
HALF_WINDOW_S = 3.0
PROXY_TO_FULL = 8
DENOISE_KERNEL = 9


def interp_nan(a):
    a = a.astype(np.float64).copy()
    bad = ~np.isfinite(a)
    if bad.any() and not bad.all():
        idx = np.arange(len(a))
        a[bad] = np.interp(idx[bad], idx[~bad], a[~bad])
    return a


def jump_speed(cx, cy, f_center, n):
    half = int(round(HALF_WINDOW_S * FPS))
    lo, hi = max(0, f_center - half), min(n, f_center + half)
    x = medfilt(cx[lo:hi], DENOISE_KERNEL) * PROXY_TO_FULL
    y = medfilt(cy[lo:hi], DENOISE_KERNEL) * PROXY_TO_FULL
    frames = np.arange(lo, hi)
    t_rel = (frames - f_center) / FPS
    vx, vy = np.gradient(x) * FPS, np.gradient(y) * FPS
    speed = np.hypot(vx, vy)
    return t_rel, speed


def main():
    proxy_dir, csv_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    only = int(sys.argv[4]) if len(sys.argv) > 4 else None
    os.makedirs(out_dir, exist_ok=True)

    tr = np.load(os.path.join(proxy_dir, "cam2_track.npz"))
    cx, cy = interp_nan(tr["cx"]), interp_nan(tr["cy"])
    n = len(cx)

    df = pd.read_csv(csv_path)
    jumps = df[df.is_jump].sort_values("t_s").reset_index(drop=True)

    all_curves = []
    for i, row in jumps.iterrows():
        jump_no = i + 1
        if only is not None and jump_no != only:
            continue
        t_center = float(row.t_s)
        f_center = int(round(t_center * FPS))
        t_rel, speed = jump_speed(cx, cy, f_center, n)
        all_curves.append((jump_no, t_center, t_rel, speed))

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(t_rel, speed, color="#1f77b4", linewidth=1.3)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.3, label="jump onset")
        ax.fill_between(t_rel, 0, speed, alpha=0.08, color="#1f77b4")
        ax.set_xlabel("time relative to jump (s)")
        ax.set_ylabel("body-centroid speed (Cam2 px/s, uncalibrated)")
        ax.set_title(f"Jump {jump_no:02d}  (t={t_center:.2f}s)  --  "
                      f"peak {speed.max():.0f} px/s at t={t_rel[np.argmax(speed)]:+.2f}s")
        ax.legend(loc="upper right")
        ax.set_xlim(-3, 3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"jump_{jump_no:02d}_speed.png"), dpi=130)
        plt.close(fig)
        print(f"[jump {jump_no:02d}] peak speed {speed.max():.0f} px/s "
              f"at t={t_rel[np.argmax(speed)]:+.2f}s", flush=True)

    if only is None and all_curves:
        fig, ax = plt.subplots(figsize=(9, 5))
        cmap = plt.get_cmap("turbo")
        grid = np.linspace(-3, 3, 601)
        stack = []
        for jump_no, t_center, t_rel, speed in all_curves:
            ax.plot(t_rel, speed, color=cmap(jump_no / len(all_curves)),
                     alpha=0.5, linewidth=0.9)
            stack.append(np.interp(grid, t_rel, speed))
        stack = np.array(stack)
        mean, sem = stack.mean(0), stack.std(0) / np.sqrt(len(stack))
        ax.plot(grid, mean, color="black", linewidth=2.2, label="mean (n=27)")
        ax.fill_between(grid, mean - sem, mean + sem, color="black", alpha=0.2)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.3, label="jump onset")
        ax.set_xlabel("time relative to jump (s)")
        ax.set_ylabel("body-centroid speed (Cam2 px/s, uncalibrated)")
        ax.set_title("Body-centroid speed, all 27 jumps aligned to jump onset")
        ax.legend()
        ax.set_xlim(-3, 3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "_overview_speed_all_jumps.png"), dpi=130)
        plt.close(fig)
        print(f"\nwrote overview -> {out_dir}/_overview_speed_all_jumps.png")


if __name__ == "__main__":
    main()
