"""
Step 4b: fully automatic labels for the training frames, for the user to
correct (not just seed -- every keypoint on every frame gets a position).

Method: SuperAnimal-Quadruped already ran on the full 600-frame clips. Per
clip we build a smooth, outlier-rejected trajectory for each relevant SA
keypoint over all 600 frames, then read off the labelled frames from that
trajectory. Temporal smoothing rides through the isolated bad frames
(motion-blur dropouts, the Cam2 head<->tail flips), which are 1-3 frame
glitches the rolling median rejects.

Mapping to the 5 project keypoints:
    snout    <- nose            (fallback upper_jaw)
    eye      <- left/right_eye   (per clip: higher median confidence)
    ear      <- matching earbase
    head_top <- midpoint(left_earbase, right_earbase)   (fallback: ear)
    nape     <- neck_base        (fallback: ear + (ear - eye))

Writes CollectedData_sem.h5/.csv per clip and a diagnostic contact sheet
in autolabel_check/<clip>.jpg.
"""
import glob
import os

import cv2
import numpy as np
import pandas as pd

CFG = open(r"E:\TreeShrew_Jumping\analysis\dlc\PROJECT_CONFIG_PATH.txt").read().strip()
PROJ = os.path.dirname(CFG)
ROOT = r"E:\TreeShrew_Jumping\analysis\dlc"
SA_DIR = os.path.join(ROOT, "superanimal")
CHECK = os.path.join(ROOT, "autolabel_check")
SCORER = "sem"
BODYPARTS = ["snout", "eye", "ear", "head_top", "nape"]
LIK_MIN = 0.35
RESID_MAX = 55.0          # px from rolling median -> treat as outlier
ROLL = 9
IMG = 1024

TRAIN_CLIPS = open(os.path.join(PROJ, "TRAIN_CLIPS.txt")).read().split()
COL = {"snout": (60, 60, 230), "eye": (60, 200, 60), "ear": (230, 130, 40),
       "head_top": (40, 220, 220), "nape": (220, 60, 200)}


def smooth_track(x, y, lik):
    """Robust smooth (x, y) over frames; returns gap-free arrays."""
    x = pd.Series(np.where(lik >= LIK_MIN, x, np.nan))
    y = pd.Series(np.where(lik >= LIK_MIN, y, np.nan))
    for _ in range(2):
        mx = x.rolling(ROLL, center=True, min_periods=2).median()
        my = y.rolling(ROLL, center=True, min_periods=2).median()
        bad = ((x - mx).abs() > RESID_MAX) | ((y - my).abs() > RESID_MAX)
        x[bad] = np.nan
        y[bad] = np.nan
    x = x.interpolate(limit_direction="both").rolling(5, center=True, min_periods=1).mean()
    y = y.interpolate(limit_direction="both").rolling(5, center=True, min_periods=1).mean()
    return x.to_numpy(), y.to_numpy()


def load_sa(stem):
    hits = glob.glob(os.path.join(SA_DIR, f"{stem}_superanimal_quadruped*.h5"))
    d = pd.read_hdf(hits[0])
    d.columns = d.columns.droplevel([0, 1])
    return d.replace(-1.0, np.nan)


def track(d, bp):
    return smooth_track(d[(bp, "x")].to_numpy(),
                        d[(bp, "y")].to_numpy(),
                        d[(bp, "likelihood")].to_numpy())


def med_lik(d, bp):
    return np.nanmedian(d[(bp, "likelihood")].to_numpy())


def build_clip(stem):
    d = load_sa(stem)
    n = len(d)
    eye_side = "left" if med_lik(d, "left_eye") >= med_lik(d, "right_eye") else "right"
    ear_side = "left" if med_lik(d, "left_earbase") >= med_lik(d, "right_earbase") else "right"

    nose = track(d, "nose")
    jaw = track(d, "upper_jaw")
    eye = track(d, f"{eye_side}_eye")
    earL, earR = track(d, "left_earbase"), track(d, "right_earbase")
    ear = earL if ear_side == "left" else earR
    neck = track(d, "neck_base")

    snout = np.where(np.isfinite(nose[0]), nose, jaw)
    # head_top: ear midpoint where both exist, else the chosen ear
    both = np.isfinite(earL[0]) & np.isfinite(earR[0])
    htx = np.where(both, (earL[0] + earR[0]) / 2, ear[0])
    hty = np.where(both, (earL[1] + earR[1]) / 2, ear[1])
    # nape: neck_base, else reflect eye across ear (roughly behind the head)
    npx = np.where(np.isfinite(neck[0]), neck[0], 2 * ear[0] - eye[0])
    npy = np.where(np.isfinite(neck[1]), neck[1], 2 * ear[1] - eye[1])

    out = {
        "snout": snout, "eye": eye, "ear": ear,
        "head_top": (htx, hty), "nape": (npx, npy),
    }
    # final clamp + fill any residual nan by nearest
    for k, (xx, yy) in out.items():
        xx = pd.Series(xx).interpolate(limit_direction="both").fillna(IMG / 2).clip(0, IMG - 1)
        yy = pd.Series(yy).interpolate(limit_direction="both").fillna(IMG / 2).clip(0, IMG - 1)
        out[k] = (xx.to_numpy(), yy.to_numpy())
    return out, eye_side, ear_side


def main():
    os.makedirs(CHECK, exist_ok=True)
    cols = pd.MultiIndex.from_product(
        [[SCORER], BODYPARTS, ["x", "y"]], names=["scorer", "bodyparts", "coords"])

    for clip in TRAIN_CLIPS:
        stem = clip[:-4]
        ld = os.path.join(PROJ, "labeled-data", stem)
        vid = os.path.join(PROJ, "videos", clip)
        tracks, es, ers = build_clip(stem)

        pngs = sorted(glob.glob(os.path.join(ld, "img*.png")))
        rows, idx, tiles = [], [], []
        cap = cv2.VideoCapture(vid)
        for png in pngs:
            fr = int(os.path.basename(png)[3:-4])
            row = []
            im = cv2.imread(png)
            for bp in BODYPARTS:
                x, y = float(tracks[bp][0][fr]), float(tracks[bp][1][fr])
                row += [x, y]
                cv2.circle(im, (int(x), int(y)), 5, COL[bp], -1)
                cv2.circle(im, (int(x), int(y)), 6, (255, 255, 255), 1)
            rows.append(row)
            idx.append(("labeled-data", stem, os.path.basename(png)))
            tiles.append(cv2.resize(im[200:750, :], (384, 268)))
        cap.release()

        data = pd.DataFrame(rows, columns=cols,
                            index=pd.MultiIndex.from_tuples(idx))
        data.to_hdf(os.path.join(ld, f"CollectedData_{SCORER}.h5"),
                    key="df_with_missing", mode="w")
        data.to_csv(os.path.join(ld, f"CollectedData_{SCORER}.csv"))

        # contact sheet
        per_row = 4
        rowsimg = [np.hstack(tiles[i:i + per_row])
                   for i in range(0, len(tiles), per_row)]
        w = max(r.shape[1] for r in rowsimg)
        rowsimg = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rowsimg]
        sheet = np.vstack(rowsimg)
        cv2.putText(sheet, f"{stem}  eye={es} ear={ers}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(CHECK, f"{stem}.jpg"), sheet)
        print(f"{stem}: {len(pngs)} frames labelled (eye={es}, ear={ers})")

    print(f"\ncontact sheets -> {CHECK}\\*.jpg")
    print("correct in napari:  double-click label.bat")


if __name__ == "__main__":
    main()
