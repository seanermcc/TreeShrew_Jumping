"""
Step 7: head position + head angle from DLC keypoints, in absolute time.

For each analysed clip:
  head_x, head_y      likelihood-weighted centroid of the confident head points
  snout_x, snout_y    snout tip
  eye_x, eye_y        eye
  head_angle_deg      orientation of the nape->snout axis in image coords
                      (0 = pointing +x / right, 90 = pointing up)
  pitch_from_eye_deg  angle of the eye->snout vector (proxy for head pitch)
  <point>_lik         per-point confidence

Frame indexing (from clips/manifest.csv):
  abs_frame = true_start_frame + clip_frame        (source .mkv frame @ 100 fps)
  t_abs_s   = abs_frame / 100
  t_rel_s   = (clip_frame - 300) / 100             (matches body3d.py convention)

Also writes clips/head3d/jump_NN_head3d.csv: a calibration-free pseudo-3D head
track combining Cam2 (X = along-tunnel, Z = height) and Cam3 (Y = across),
same recipe as body3d.py but for the head instead of the body centroid.
"""
import glob
import os

import numpy as np
import pandas as pd

CFG = open(r"E:\TreeShrew_Jumping\analysis\dlc\PROJECT_CONFIG_PATH.txt").read().strip()
PROJ = os.path.dirname(CFG)
ROOT = r"E:\TreeShrew_Jumping\analysis\dlc"
VID_DIR = os.path.join(PROJ, "videos")
OUT = os.path.join(ROOT, "head_pose")
OUT3D = os.path.join(ROOT, "head3d")
PCUTOFF = 0.6
FPS = 100.0
IMG_H = 1024
HEAD_PTS = ["eye", "ear", "head_top", "nape"]


def load_pred(clip_stem):
    hits = sorted(glob.glob(os.path.join(VID_DIR, clip_stem + "DLC*_filtered.h5"))) \
        or sorted(glob.glob(os.path.join(VID_DIR, clip_stem + "DLC*.h5")))
    if not hits:
        return None
    df = pd.read_hdf(hits[0])
    df.columns = df.columns.droplevel(0)  # drop scorer
    return df


def xyp(df, bp):
    return (df[(bp, "x")].to_numpy(),
            df[(bp, "y")].to_numpy(),
            df[(bp, "likelihood")].to_numpy())


def head_pose(df):
    n = len(df)
    sx, sy, sp = xyp(df, "snout")
    ex, ey, ep = xyp(df, "eye")
    npx, npy, npp = xyp(df, "nape")

    # weighted head centroid
    hx = np.full(n, np.nan); hy = np.full(n, np.nan)
    for i in range(n):
        xs, ys, ws = [], [], []
        for bp in HEAD_PTS:
            x, y, p = df[(bp, "x")][i], df[(bp, "y")][i], df[(bp, "likelihood")][i]
            if p >= PCUTOFF:
                xs.append(x); ys.append(y); ws.append(p)
        if ws:
            w = np.array(ws)
            hx[i] = np.dot(xs, w) / w.sum()
            hy[i] = np.dot(ys, w) / w.sum()

    def ang(ax, ay, ap, bx, by, bp_, thr=PCUTOFF):
        ok = (ap >= thr) & (bp_ >= thr)
        a = np.full(n, np.nan)
        dx = bx - ax
        dy = -(by - ay)  # image y grows down -> flip so +y is up
        a[ok] = np.degrees(np.arctan2(dy[ok], dx[ok]))
        return a

    head_angle = ang(npx, npy, npp, sx, sy, sp)      # nape -> snout
    pitch_eye = ang(ex, ey, ep, sx, sy, sp)          # eye  -> snout

    return pd.DataFrame(dict(
        head_x=hx, head_y=hy, head_y_up=IMG_H - 1 - hy,
        snout_x=sx, snout_y=sy, snout_lik=sp,
        eye_x=ex, eye_y=ey, eye_lik=ep,
        nape_x=npx, nape_y=npy, nape_lik=npp,
        head_angle_deg=head_angle, pitch_from_eye_deg=pitch_eye,
    ))


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(OUT3D, exist_ok=True)
    man = pd.read_csv(os.path.join(ROOT, "clips", "manifest.csv"))
    man["stem"] = man["clip"].str.replace(".mp4", "", regex=False)

    per_clip = {}
    for _, m in man.iterrows():
        stem = m["stem"]
        df = load_pred(stem)
        if df is None:
            print(f"  no predictions for {stem} (run 06_analyze.py first)")
            continue
        hp = head_pose(df)
        cf = np.arange(len(hp))
        hp.insert(0, "clip_frame", cf)
        hp.insert(1, "abs_frame", int(m["true_start_frame"]) + cf)
        hp.insert(2, "t_abs_s", hp["abs_frame"] / FPS)
        hp.insert(3, "t_rel_s", (cf - 300) / FPS)
        hp.insert(4, "cam", int(m["cam"]))
        hp.insert(5, "jump_no", int(m["jump_no"]))
        hp.to_csv(os.path.join(OUT, f"{stem}_headpose.csv"), index=False)
        per_clip[(int(m["jump_no"]), int(m["cam"]))] = hp
        good = np.isfinite(hp["head_x"]).mean() * 100
        print(f"{stem}: head found {good:.0f}% of frames")

    # combined
    if per_clip:
        pd.concat(per_clip.values(), ignore_index=True).to_csv(
            os.path.join(OUT, "_all_headpose.csv"), index=False)

    # pseudo-3D head track (Cam2 = X,Z ; Cam3 = Y), body3d.py recipe
    for jump in sorted({j for (j, _) in per_clip}):
        c2, c3 = per_clip.get((jump, 2)), per_clip.get((jump, 3))
        if c2 is None or c3 is None:
            continue
        d = pd.DataFrame(dict(
            clip_frame=c2["clip_frame"], t_rel_s=c2["t_rel_s"], t_abs_s=c2["t_abs_s"],
            X_along=c2["head_x"], Z_up=c2["head_y_up"], Y_across=c3["head_x"],
            head_angle_cam2_deg=c2["head_angle_deg"],
            head_angle_cam3_deg=c3["head_angle_deg"],
        ))
        d.to_csv(os.path.join(OUT3D, f"jump_{jump:02d}_head3d.csv"), index=False)

    print(f"\nwrote {OUT}\\*_headpose.csv  and  {OUT3D}\\jump_*_head3d.csv")


if __name__ == "__main__":
    main()
