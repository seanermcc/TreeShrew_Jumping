"""
Step 4: extract frames to label + seed them with SuperAnimal predictions.

Frame selection per training clip (600 frames, jump ~frame 300):
  - kmeans frames across the whole clip (diverse resting/approach poses)
  - forced frames through the ballistic window (takeoff -> apex -> landing),
    where zero-shot SuperAnimal is least reliable and hand labels matter most

Seeding: map SuperAnimal-Quadruped predictions to the 5 project keypoints
  snout <- nose (fallback upper_jaw)
  eye   <- left/right_eye  (per-clip, whichever side is more confident)
  ear   <- matching earbase
  head_top <- midpoint(left_earbase, right_earbase)   [rough seed]
  nape  <- neck_base
Only points with SA likelihood > MIN_SEED are written; the rest are left
empty (NaN) for the labeller to place. Seeds are a starting point only --
every frame must still be checked by hand.
"""
import glob
import os

import cv2
import numpy as np
import pandas as pd

import deeplabcut

CFG = open(r"E:\TreeShrew_Jumping\analysis\dlc\PROJECT_CONFIG_PATH.txt").read().strip()
PROJ = os.path.dirname(CFG)
ROOT = r"E:\TreeShrew_Jumping\analysis\dlc"
SA_DIR = os.path.join(ROOT, "superanimal")
SCORER = "sem"
BODYPARTS = ["snout", "eye", "ear", "head_top", "nape"]
MIN_SEED = 0.5
FORCE_FRAMES = [250, 275, 290, 300, 310, 325, 350]  # ballistic window
N_KMEANS = 6

TRAIN_CLIPS = open(os.path.join(PROJ, "TRAIN_CLIPS.txt")).read().split()


def sa_h5(stem):
    hits = glob.glob(os.path.join(SA_DIR, f"{stem}_superanimal_quadruped*.h5"))
    return pd.read_hdf(hits[0]) if hits else None


def seed_row(df, frame, clip_eye_side, clip_ear_side):
    """Return dict {(bp,coord): value} for one frame, from SA predictions."""
    if df is None or frame >= len(df):
        return {}
    r = df.iloc[frame]
    r.index = r.index.droplevel([0, 1])  # -> (bodypart, coord)

    def get(bp, thr=MIN_SEED):
        try:
            x, y, p = r[(bp, "x")], r[(bp, "y")], r[(bp, "likelihood")]
            return (float(x), float(y)) if p >= thr else None
        except KeyError:
            return None

    # Guard against the head<->tail collapse SuperAnimal shows on the most
    # motion-blurred Cam2 frames: if the predicted nose sits right on top of
    # the predicted tail base, the whole head cluster is untrustworthy -->
    # seed nothing for this frame (a blank frame is better than a wrong seed).
    nose_g = get("nose", 0.0)
    tail_g = get("tail_base", 0.6)
    if nose_g and tail_g:
        d = np.hypot(nose_g[0] - tail_g[0], nose_g[1] - tail_g[1])
        if d < 220:
            return {}

    out = {}
    snout = get("nose") or get("upper_jaw")
    eye = get(f"{clip_eye_side}_eye")
    ear = get(f"{clip_ear_side}_earbase")
    nape = get("neck_base")
    le, re = get("left_earbase"), get("right_earbase")
    head_top = None
    if le and re:
        head_top = ((le[0] + re[0]) / 2, (le[1] + re[1]) / 2)
    elif ear:
        head_top = ear
    for bp, xy in [("snout", snout), ("eye", eye), ("ear", ear),
                   ("head_top", head_top), ("nape", nape)]:
        if xy is not None:
            out[(bp, "x")], out[(bp, "y")] = xy
    return out


def best_side(df):
    if df is None:
        return "left", "left"
    d = df.copy()
    d.columns = d.columns.droplevel([0, 1])
    eye = "left" if d[("left_eye", "likelihood")].mean() >= d[("right_eye", "likelihood")].mean() else "right"
    ear = "left" if d[("left_earbase", "likelihood")].mean() >= d[("right_earbase", "likelihood")].mean() else "right"
    return eye, ear


def main():
    import sys
    seed_only = "--seed-only" in sys.argv

    # 1) kmeans extraction (DLC handles PNG naming + folders)
    if not seed_only:
        deeplabcut.extract_frames(
            CFG, mode="automatic", algo="kmeans", userfeedback=False, crop=False,
            videos_list=[os.path.join(PROJ, "videos", c) for c in TRAIN_CLIPS],
        )

    cols = pd.MultiIndex.from_product(
        [[SCORER], BODYPARTS, ["x", "y"]], names=["scorer", "bodyparts", "coords"])

    for clip in TRAIN_CLIPS:
        stem = clip[:-4]
        vid = os.path.join(PROJ, "videos", clip)
        ld = os.path.join(PROJ, "labeled-data", stem)
        os.makedirs(ld, exist_ok=True)
        df = sa_h5(stem)
        eye_side, ear_side = best_side(df)

        # 2) force ballistic-window frames as PNGs
        cap = cv2.VideoCapture(vid)
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 600
        pad = len(str(n_total))  # DLC's img-name zero-padding
        for fr in FORCE_FRAMES:
            png = os.path.join(ld, f"img{fr:0{pad}d}.png")
            if not os.path.exists(png):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
                ok, im = cap.read()
                if ok:
                    cv2.imwrite(png, im)
        cap.release()

        # 3) build CollectedData seed for every PNG in the folder
        pngs = sorted(glob.glob(os.path.join(ld, "img*.png")))
        rows, idx = [], []
        for png in pngs:
            fr = int(os.path.basename(png)[3:-4])
            seed = seed_row(df, fr, eye_side, ear_side)
            rows.append([seed.get((bp, c), np.nan) for bp in BODYPARTS for c in ("x", "y")])
            idx.append(("labeled-data", stem, os.path.basename(png)))
        data = pd.DataFrame(rows, columns=cols,
                            index=pd.MultiIndex.from_tuples(idx))
        data.to_hdf(os.path.join(ld, f"CollectedData_{SCORER}.h5"),
                    key="df_with_missing", mode="w")
        data.to_csv(os.path.join(ld, f"CollectedData_{SCORER}.csv"))
        seeded = np.isfinite(data.values).sum() // 2
        print(f"{stem}: {len(pngs)} frames, eye={eye_side} ear={ear_side}, "
              f"{seeded}/{len(pngs) * 5} keypoints seeded")

    print("\nNext: label/correct in napari ->  deeplabcut.label_frames(CFG)")


if __name__ == "__main__":
    main()
