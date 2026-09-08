"""
Calibration-free, body-level 3D trajectory for each jump, from centroid tracks.

No camera calibration exists for this rig, so this is NOT metric 3D -- it is a
pseudo-3D reconstruction from two roughly orthogonal views, a standard
qualitative technique when calibration is unavailable:

  X (along-tunnel)  <- Cam2 centroid x   (Cam2 is a clean side profile)
  Z (height, up)     <- Cam2 centroid y_up
  Y (across-tunnel) <- Cam4 centroid x   (Cam4 looks down the tunnel axis,
                                           roughly orthogonal to Cam2)

Units are full-resolution pixels (proxy px * 8); there is no cm/mm scale
without a calibration target. Treat shapes/relative magnitudes as meaningful,
absolute distances as not.

For each jump: 3 s before to 3 s after the detection frame (matches the video
clips), one CSV of the trajectory, one 3D plot. Plus two summary figures
across all 27 jumps.
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
# Suppresses single/few-frame blob-tracker glitches (glare, edge reflections)
# without smearing the real jump excursion, which spans tens of frames.
# Checked against the raw signal: <0.1% of frames deviate from a 9-frame
# median by >20 px, so this is cleaning outliers, not the jump itself.
DENOISE_KERNEL = 9


def interp_nan(a):
    a = a.astype(np.float64).copy()
    bad = ~np.isfinite(a)
    if bad.any() and not bad.all():
        idx = np.arange(len(a))
        a[bad] = np.interp(idx[bad], idx[~bad], a[~bad])
    return a


def load_track(proxy_dir, cam):
    d = np.load(os.path.join(proxy_dir, f"cam{cam}_track.npz"))
    return interp_nan(d["cx"]), interp_nan(d["y_up"])


def main():
    proxy_dir, csv_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)

    cam2_cx, cam2_yup = load_track(proxy_dir, 2)
    cam4_cx, _ = load_track(proxy_dir, 4)
    n = min(len(cam2_cx), len(cam4_cx))

    df = pd.read_csv(csv_path)
    jumps = df[df.is_jump].sort_values("t_s").reset_index(drop=True)

    all_traj = []  # for the overview plots
    for i, row in jumps.iterrows():
        jump_no = i + 1
        t_center = float(row.t_s)
        f_center = int(round(t_center * FPS))
        half = int(round(HALF_WINDOW_S * FPS))
        lo, hi = max(0, f_center - half), min(n, f_center + half)

        frames = np.arange(lo, hi)
        t_rel = (frames - f_center) / FPS
        x_raw = cam2_cx[lo:hi] * PROXY_TO_FULL
        z_raw = cam2_yup[lo:hi] * PROXY_TO_FULL
        y_raw = cam4_cx[lo:hi] * PROXY_TO_FULL

        k = min(DENOISE_KERNEL, len(x_raw) - (1 - len(x_raw) % 2))  # odd, <= window
        if k >= 3:
            x, y, z = (medfilt(a, k) for a in (x_raw, y_raw, z_raw))
        else:
            x, y, z = x_raw, y_raw, z_raw

        vx, vy, vz = (np.gradient(a) * FPS for a in (x, y, z))
        speed3d = np.sqrt(vx**2 + vy**2 + vz**2)

        out = pd.DataFrame({
            "frame": frames, "t_rel_s": t_rel, "t_abs_s": frames / FPS,
            "x_px": x, "y_px": y, "z_px": z, "speed3d_px_s": speed3d,
            "x_px_raw": x_raw, "y_px_raw": y_raw, "z_px_raw": z_raw,
        })
        out.to_csv(os.path.join(out_dir, f"jump_{jump_no:02d}_traj.csv"), index=False)
        all_traj.append((jump_no, t_rel, x, y, z))

        # Per-jump 3D plot
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(x, y, z, c=t_rel, cmap="viridis", s=6)
        ax.plot(x, y, z, color="gray", alpha=0.3, linewidth=1)
        k0 = int(np.argmin(np.abs(t_rel)))
        ax.scatter(*[[a] for a in (x[0], y[0], z[0])], color="green", s=60, label="t=-3s")
        ax.scatter(*[[a] for a in (x[k0], y[k0], z[k0])], color="red", s=80, label="jump")
        ax.scatter(*[[a] for a in (x[-1], y[-1], z[-1])], color="blue", s=60, label="t=+3s")
        ax.set_xlabel("X: along-tunnel (px, Cam2)")
        ax.set_ylabel("Y: across-tunnel (px, Cam4)")
        ax.set_zlabel("Z: height (px, Cam2)")
        ax.set_title(f"Jump {jump_no:02d}  (t={t_center:.2f}s)  -- pseudo-3D, uncalibrated")
        ax.legend(fontsize=8)
        fig.colorbar(sc, ax=ax, shrink=0.6, label="t rel. to jump (s)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"jump_{jump_no:02d}_3d.png"), dpi=120)
        plt.close(fig)
        print(f"[jump {jump_no:02d}] wrote CSV + 3D plot ({hi-lo} frames)", flush=True)

    # Overview 1: all trajectories overlaid in 3D
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("turbo")
    for jump_no, t_rel, x, y, z in all_traj:
        ax.plot(x, y, z, color=cmap(jump_no / len(all_traj)), alpha=0.7, linewidth=1)
    ax.set_xlabel("X: along-tunnel (px)")
    ax.set_ylabel("Y: across-tunnel (px)")
    ax.set_zlabel("Z: height (px)")
    ax.set_title("All 27 jumps, pseudo-3D centroid trajectories (uncalibrated)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "_overview_3d_all_jumps.png"), dpi=130)
    plt.close(fig)

    # Overview 2: height profile aligned to jump time -- most direct answer to
    # "how does the (body, here) move before/during/after the jump"
    fig, ax = plt.subplots(figsize=(9, 5))
    for jump_no, t_rel, x, y, z in all_traj:
        ax.plot(t_rel, z, color=cmap(jump_no / len(all_traj)), alpha=0.7, linewidth=1)
    ax.axvline(0, color="k", linestyle="--", linewidth=1, label="jump (detection frame)")
    ax.set_xlabel("time relative to jump (s)")
    ax.set_ylabel("height, Z (px, Cam2)")
    ax.set_title("Body height across all 27 jumps, aligned to jump time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "_overview_height_vs_time.png"), dpi=130)
    plt.close(fig)

    print(f"\n{len(jumps)} jumps done -> {out_dir}")


if __name__ == "__main__":
    main()
