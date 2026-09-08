"""
Build a jump inventory from the tracks + motion-energy traces.

Motion energy reliably flags the moments something fast happened (verified by
eye on the t=831 s event). That alone does not separate a jump from vigorous
grooming, so each candidate is scored with kinematics from the side view: a jump
shows a large, smooth excursion of the body centroid with a ballistic vertical
profile, while grooming stays put.

Thresholds are deliberately explicit at the top so they can be tuned against
the contact sheets rather than buried in the logic.
"""
import os
import sys

import numpy as np

FPS = 100.0
PROXY_TO_FULL = 8          # 128 px proxy -> 1024 px original

MOTION_PCT = 99.0          # candidate peaks above this percentile
REFRACTORY = 150           # 1.5 s minimum between distinct events
WINDOW = 150               # +/- 1.5 s of track examined per candidate
# Peak centroid speed cleanly separates ballistic flight from everything else:
# the observed distribution has a wide empty gap between 1000 and 2630 px/s,
# with locomotion/grooming below and jumps above. Verified against contact
# sheets at t=831 s (fastest) and t=894 s (just above the cut).
MIN_PEAK_SPEED = 1815.0    # proxy px/s


def interp_nan(a):
    a = a.copy()
    bad = ~np.isfinite(a)
    if bad.all():
        return a
    idx = np.arange(len(a))
    a[bad] = np.interp(idx[bad], idx[~bad], a[~bad])
    return a


def candidates(motion, pct=MOTION_PCT, refractory=REFRACTORY):
    thr = np.percentile(motion, pct)
    order = np.argsort(motion)[::-1]
    picked = []
    for i in order:
        if motion[i] < thr:
            break
        if all(abs(int(i) - p) > refractory for p in picked):
            picked.append(int(i))
    return sorted(picked), thr


def kinematics(cx, cy, y_up, area, centre, window=WINDOW):
    lo, hi = max(0, centre - window), min(len(cx), centre + window)
    x, y, yu = cx[lo:hi], cy[lo:hi], y_up[lo:hi]
    if len(x) < 10:
        return None
    dx = np.nanmax(x) - np.nanmin(x)
    dy = np.nanmax(yu) - np.nanmin(yu)
    disp = float(np.hypot(dx, dy))
    vx = np.gradient(x) * FPS
    vy = np.gradient(yu) * FPS
    speed = np.hypot(vx, vy)
    return {
        "disp_px": disp,
        "dx_px": float(dx),
        "dy_px": float(dy),
        "peak_speed_px_s": float(np.nanmax(speed)),
        "apex_y_up": float(np.nanmax(yu)),
        "start_y_up": float(yu[0]),
        "end_y_up": float(yu[-1]),
        "net_height_change": float(yu[-1] - yu[0]),
        "median_area": float(np.nanmedian(area[lo:hi])),
    }


def main():
    proxy_dir = sys.argv[1]
    det_cam = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    motion = np.load(os.path.join(proxy_dir, f"cam{det_cam}_signals.npz"))["motion_energy"]
    tr = np.load(os.path.join(proxy_dir, f"cam{det_cam}_track.npz"))
    cx, cy = interp_nan(tr["cx"]), interp_nan(tr["cy"])
    y_up, area = interp_nan(tr["y_up"]), tr["area"]

    ev, thr = candidates(motion)
    print(f"Detection camera: Cam{det_cam}")
    print(f"Motion threshold (p{MOTION_PCT}) = {thr:.3f}")
    print(f"{len(ev)} candidate events\n")

    rows = []
    for i in ev:
        k = kinematics(cx, cy, y_up, area, i)
        if k is None:
            continue
        k["frame"] = i
        k["t_s"] = i / FPS
        k["motion_peak"] = float(motion[i])
        k["is_jump"] = k["peak_speed_px_s"] >= MIN_PEAK_SPEED
        rows.append(k)

    jumps = [r for r in rows if r["is_jump"]]
    print(f"{len(jumps)} of {len(rows)} candidates exceed "
          f"{MIN_PEAK_SPEED:.0f} px/s peak speed\n")
    hdr = (f"{'t (s)':>9} {'frame':>7} {'motion':>7} {'disp':>7} "
           f"{'dx':>7} {'dy':>7} {'peak v':>9} {'dHeight':>8}  jump")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['t_s']:9.2f} {r['frame']:7d} {r['motion_peak']:7.2f} "
              f"{r['disp_px']:7.1f} {r['dx_px']:7.1f} {r['dy_px']:7.1f} "
              f"{r['peak_speed_px_s']:9.1f} {r['net_height_change']:8.1f}"
              f"  {'YES' if r['is_jump'] else '.'}")

    import csv
    out = os.path.join(proxy_dir, f"jump_inventory_cam{det_cam}.csv")
    if rows:
        keys = ["t_s", "frame", "motion_peak", "disp_px", "dx_px", "dy_px",
                "peak_speed_px_s", "apex_y_up", "start_y_up", "end_y_up",
                "net_height_change", "median_area", "is_jump"]
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in keys})
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
