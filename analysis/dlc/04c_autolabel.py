"""
Step 4c: pick jump-phase frames AND auto-label them (supersedes 04 / 04b).

Frame pick per clip: forced ballistic frames (250-350) + evenly spaced frames
drawn only from where SuperAnimal actually sees the animal, restricted to the
jump-relevant window [150, 480]. No empty-arena frames.

Auto-label: SuperAnimal-Quadruped predictions, robust-smoothed with a SHORT
temporal window (+/-3 frames) inside each contiguous "animal present" run so
1-2 frame glitches (blur dropouts, Cam2 head<->tail flips) are rejected but
real head motion is kept. No long interpolation across absences.

    snout    <- nose (fallback upper_jaw)
    eye      <- left/right_eye  (per clip, higher median confidence)
    ear      <- matching earbase
    head_top <- midpoint(earbases), fallback single ear nudged off the eye axis
    nape     <- neck_base, fallback ear + (ear - eye)

Keypoints with no usable SuperAnimal support are left EMPTY (NaN) for the
labeller. Diagnostic contact sheet per clip -> autolabel_check/<clip>.jpg.
"""
import glob
import os
import shutil

import cv2
import numpy as np
import pandas as pd

CFG = open(r"E:\TreeShrew_Jumping\analysis\dlc\PROJECT_CONFIG_PATH.txt").read().strip()
PROJ = os.path.dirname(CFG)
ROOT = r"E:\TreeShrew_Jumping\analysis\dlc"
SA_DIR = os.path.join(ROOT, "superanimal")
CHECK = os.path.join(ROOT, "autolabel_check")
SCORER = "sem"
BP = ["snout", "eye", "ear", "head_top", "nape"]
IMG = 1024
LIK = 0.4
FORCED = [250, 265, 280, 290, 300, 310, 320, 335, 350]
N_SPREAD = 6
WIN = (150, 480)
COL = {"snout": (60, 60, 230), "eye": (60, 200, 60), "ear": (230, 130, 40),
       "head_top": (40, 220, 220), "nape": (220, 60, 200)}

TRAIN_CLIPS = open(os.path.join(PROJ, "TRAIN_CLIPS.txt")).read().split()


def load_sa(stem):
    h = glob.glob(os.path.join(SA_DIR, f"{stem}_superanimal_quadruped*.h5"))[0]
    d = pd.read_hdf(h)
    d.columns = d.columns.droplevel([0, 1])
    return d.replace(-1.0, np.nan)


def runs(mask):
    """Yield (start, end) inclusive index ranges of True runs."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return
    splits = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[splits + 1]]
    ends = np.r_[idx[splits], idx[-1]]
    yield from zip(starts, ends)


def smooth_bp(d, bp, present):
    x = d[(bp, "x")].to_numpy().copy()
    y = d[(bp, "y")].to_numpy().copy()
    lk = d[(bp, "likelihood")].to_numpy()
    ok = np.isfinite(lk) & (lk >= 0.3)
    xs = np.full(len(x), np.nan)
    ys = np.full(len(x), np.nan)
    for a, b in runs(present):
        sl = slice(a, b + 1)
        sx = pd.Series(np.where(ok[sl], x[sl], np.nan))
        sy = pd.Series(np.where(ok[sl], y[sl], np.nan))
        for _ in range(3):
            mx = sx.rolling(11, center=True, min_periods=2).median()
            my = sy.rolling(11, center=True, min_periods=2).median()
            bad = ((sx - mx).abs() > 38) | ((sy - my).abs() > 38)
            # also drop points that are a there-and-back spike vs both neighbours
            dxf, dxb = sx.diff(), sx.diff(-1)
            dyf, dyb = sy.diff(), sy.diff(-1)
            spike = ((dxf.abs() > 40) & (dxb.abs() > 40) & (np.sign(dxf) == np.sign(dxb))) | \
                    ((dyf.abs() > 40) & (dyb.abs() > 40) & (np.sign(dyf) == np.sign(dyb)))
            sx[bad | spike] = np.nan
            sy[bad | spike] = np.nan
        sx = sx.interpolate(limit=8, limit_area="inside")
        sy = sy.interpolate(limit=8, limit_area="inside")
        sx = sx.rolling(3, center=True, min_periods=1).mean()
        sy = sy.rolling(3, center=True, min_periods=1).mean()
        xs[sl] = sx.to_numpy()
        ys[sl] = sy.to_numpy()
    return xs, ys


def med_lik(d, bp):
    v = d[(bp, "likelihood")].to_numpy()
    v = v[np.isfinite(v)]
    return np.median(v) if len(v) else 0.0


def build(stem):
    d = load_sa(stem)
    n = len(d)
    stack = np.c_[d[("nose", "likelihood")], d[("left_eye", "likelihood")],
                  d[("right_eye", "likelihood")]]
    with np.errstate(all="ignore"):
        head_lik = np.where(np.isnan(stack).all(axis=1), np.nan, np.nanmax(stack, axis=1))
    present = np.isfinite(head_lik) & (head_lik >= LIK)

    es = "left" if med_lik(d, "left_eye") >= med_lik(d, "right_eye") else "right"
    ers = "left" if med_lik(d, "left_earbase") >= med_lik(d, "right_earbase") else "right"

    nose = smooth_bp(d, "nose", present)
    jaw = smooth_bp(d, "upper_jaw", present)
    eye = list(smooth_bp(d, f"{es}_eye", present))
    earL = smooth_bp(d, "left_earbase", present)
    earR = smooth_bp(d, "right_earbase", present)
    ear = list(earL if ers == "left" else earR)
    neck = smooth_bp(d, "neck_base", present)
    back = smooth_bp(d, "back_base", present)
    tail = smooth_bp(d, "tail_base", present)

    snout = [np.where(np.isfinite(nose[0]), nose[0], jaw[0]),
             np.where(np.isfinite(nose[1]), nose[1], jaw[1])]
    both = np.isfinite(earL[0]) & np.isfinite(earR[0])
    ht = [np.where(both, (earL[0] + earR[0]) / 2, ear[0]),
          np.where(both, (earL[1] + earR[1]) / 2, ear[1])]
    nape = [np.where(np.isfinite(neck[0]), neck[0], 2 * ear[0] - eye[0]),
            np.where(np.isfinite(neck[1]), neck[1], 2 * ear[1] - eye[1])]

    # head<->tail flip correction: headward body axis is (back_base - tail_base);
    # if the snout projects onto the tail side of back_base, mirror the whole
    # head cluster through back_base.
    bx, by = back[0] - tail[0], back[1] - tail[1]
    proj = (snout[0] - back[0]) * bx + (snout[1] - back[1]) * by
    flip = np.isfinite(proj) & (proj < 0)
    for pt in (snout, eye, ear, ht, nape):
        pt[0] = np.where(flip, 2 * back[0] - pt[0], pt[0])
        pt[1] = np.where(flip, 2 * back[1] - pt[1], pt[1])

    tracks = {"snout": snout, "eye": eye, "ear": ear, "head_top": ht, "nape": nape}
    # a smoothed point that lands on/over the frame edge is really out of view
    # -> leave it EMPTY for the labeller rather than pinning it to the border
    m = 26
    out = {}
    for k, (xx, yy) in tracks.items():
        xx = np.asarray(xx, float); yy = np.asarray(yy, float)
        oob = (xx < m) | (xx > IMG - 1 - m) | (yy < m) | (yy > IMG - 1 - m)
        xx[oob] = np.nan; yy[oob] = np.nan
        out[k] = (xx, yy)
    return out, present, es, ers, d


def pick_frames(present):
    """Frames to label: present frames spread across the jump window, plus
    forced ballistic frames that are close to a present frame (animal is there,
    just blurred). Forced frames far from any detection = animal out of view."""
    pres_idx = np.where(present)[0]
    pool = [i for i in pres_idx if WIN[0] <= i <= WIN[1]]
    picks = set()
    if pool:
        picks.update(int(pool[i]) for i in
                     np.linspace(0, len(pool) - 1, min(N_SPREAD, len(pool))).astype(int))
    for f in FORCED:
        if 0 <= f < len(present) and len(pres_idx) and np.min(np.abs(pres_idx - f)) <= 25:
            picks.add(f)
    return sorted(picks)


def main():
    os.makedirs(CHECK, exist_ok=True)
    cols = pd.MultiIndex.from_product([[SCORER], BP, ["x", "y"]],
                                      names=["scorer", "bodyparts", "coords"])
    total = 0
    for clip in TRAIN_CLIPS:
        stem = clip[:-4]
        ld = os.path.join(PROJ, "labeled-data", stem)
        vid = os.path.join(PROJ, "videos", clip)
        if os.path.isdir(ld):
            shutil.rmtree(ld)
        os.makedirs(ld)

        tracks, present, es, ers, _ = build(stem)
        frames = pick_frames(present)
        pad = len(str(int(cv2.VideoCapture(vid).get(cv2.CAP_PROP_FRAME_COUNT)) or 600))

        cap = cv2.VideoCapture(vid)
        rows, idx, tiles = [], [], []
        for fr in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
            okr, im = cap.read()
            if not okr:
                continue
            name = f"img{fr:0{pad}d}.png"
            cv2.imwrite(os.path.join(ld, name), im)
            row, vis = [], im.copy()
            cx = cy = None
            for bp in BP:
                x, y = tracks[bp][0][fr], tracks[bp][1][fr]
                if np.isfinite(x) and np.isfinite(y):
                    row += [float(x), float(y)]
                    cv2.circle(vis, (int(x), int(y)), 5, COL[bp], -1)
                    cv2.circle(vis, (int(x), int(y)), 6, (255, 255, 255), 1)
                    cx, cy = x, y
                else:
                    row += [np.nan, np.nan]
            rows.append(row)
            idx.append(("labeled-data", stem, name))
            if cx is None:
                cx, cy = IMG / 2, IMG / 2
            x0, y0 = int(np.clip(cx - 190, 0, IMG - 380)), int(np.clip(cy - 140, 0, IMG - 280))
            tiles.append(cv2.resize(vis[y0:y0 + 280, x0:x0 + 380], (380, 280)))

        cap.release()
        data = pd.DataFrame(rows, columns=cols, index=pd.MultiIndex.from_tuples(idx))
        data.to_hdf(os.path.join(ld, f"CollectedData_{SCORER}.h5"),
                    key="df_with_missing", mode="w")
        data.to_csv(os.path.join(ld, f"CollectedData_{SCORER}.csv"))

        pr = 4
        rws = [np.hstack(tiles[i:i + pr]) for i in range(0, len(tiles), pr)]
        w = max(r.shape[1] for r in rws)
        rws = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rws]
        sheet = np.vstack(rws)
        filled = int(np.isfinite(data.values).sum() // 2)
        cv2.putText(sheet, f"{stem}  eye={es} ear={ers}  {filled}/{len(rows) * 5} pts",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(CHECK, f"{stem}.jpg"), sheet)
        total += len(rows)
        print(f"{stem}: {len(rows)} frames, {filled}/{len(rows)*5} pts filled (eye={es} ear={ers})")

    print(f"\n{total} frames total.  contact sheets -> {CHECK}")


if __name__ == "__main__":
    main()
