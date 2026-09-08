"""
Shared plumbing for the v3 jump analysis.

Everything downstream reads from ONE master keypoint table built here from the
raw DeepLabCut predictions (all 5 head keypoints, both cameras, indexed by the
absolute frame number of the original 20260730_CamN.mkv), plus the body-centroid
track from the proxy pipeline and the static arena geometry.

Units: full-resolution image pixels (1024x1024) and px/s. There is no camera
calibration for this rig, so nothing here is metric -- relative magnitudes and
shapes are meaningful, absolute distances are not.

Image y grows DOWN. `*_y_up` = 1023 - y, so larger = physically higher.
"""
import os
import re
import glob
import numpy as np
import pandas as pd
from scipy.signal import medfilt, savgol_filter

FPS = 100.0
AN = os.path.dirname(os.path.abspath(__file__))
DLC = os.path.join(AN, "dlc")
VIDEOS = os.path.join(DLC, "TreeShrewHead-sem-2026-08-27", "videos")
PROXY = os.path.join(AN, "proxy")
SRC = os.path.join(os.path.dirname(AN), "JumpingTest_v6_07302026", "20260730_Cam{}.mkv")
FFMPEG = os.path.join(DLC, "py310", "Lib", "site-packages",
                      "imageio_ffmpeg", "binaries", "ffmpeg-win-x86_64-v7.1.exe")
FONT = r"C\:/Windows/Fonts/arial.ttf"

V3 = os.path.join(AN, "v3")
KEYPOINT_CACHE = os.path.join(V3, "_keypoints_master.csv")

BODYPARTS = ["snout", "eye", "ear", "head_top", "nape"]
# The head centroid deliberately EXCLUDES the snout (matches 07_head_pose.py):
# the snout is the tip of a lever and swings with pitch, so including it would
# make "head position" move whenever the animal only re-aimed its head.
HEAD_PTS = ["eye", "ear", "head_top", "nape"]
PCUTOFF = 0.6          # same gate as 07_head_pose.py
PROXY_TO_FULL = 8      # 128 px proxy -> 1024 px original
IMG = 1024

# per-clip window start frames, from dlc/clips/manifest.csv (clip_frame 0 = this)
MANIFEST = os.path.join(DLC, "clips", "manifest.csv")


# --------------------------------------------------------------------------- #
# master keypoint table
# --------------------------------------------------------------------------- #
def build_keypoints(force=False):
    """abs_frame x cam -> all 5 keypoints (x, y, likelihood).

    The 27 six-second DLC windows overlap in time. Where two clips cover the
    same absolute frame we keep the sample from whichever clip has that frame
    closest to its centre -- the animal is best framed there, and it keeps the
    choice deterministic rather than order-dependent.
    """
    os.makedirs(V3, exist_ok=True)
    if os.path.exists(KEYPOINT_CACHE) and not force:
        return pd.read_csv(KEYPOINT_CACHE)

    man = pd.read_csv(MANIFEST)
    start = {(int(r.cam), int(r.jump_no)): int(r.true_start_frame)
             for r in man.itertuples()}

    rows = []
    for h5 in sorted(glob.glob(os.path.join(VIDEOS, "*_filtered.h5"))):
        base = os.path.basename(h5)
        m = re.match(r"cam(\d)_jump_(\d+)", base)   # cam2_jump_07DLC_...h5
        cam, jump = int(m.group(1)), int(m.group(2))
        d = pd.read_hdf(h5)
        d.columns = d.columns.droplevel(0)      # drop scorer level
        n = len(d)
        out = pd.DataFrame({
            "abs_frame": start[(cam, jump)] + np.arange(n),
            "clip_frame": np.arange(n),
            "cam": cam, "jump_no": jump,
        })
        for bp in BODYPARTS:
            out[f"{bp}_x"] = d[(bp, "x")].to_numpy()
            out[f"{bp}_y"] = d[(bp, "y")].to_numpy()
            out[f"{bp}_lik"] = d[(bp, "likelihood")].to_numpy()
        rows.append(out)

    df = pd.concat(rows, ignore_index=True)
    df["central"] = (df["clip_frame"] - 300).abs()
    df = (df.sort_values(["cam", "abs_frame", "central"])
            .drop_duplicates(["cam", "abs_frame"], keep="first")
            .drop(columns="central")
            .sort_values(["cam", "abs_frame"])
            .reset_index(drop=True))

    # gate on likelihood, then derive the head centroid the same way 07 did
    for bp in BODYPARTS:
        bad = df[f"{bp}_lik"] < PCUTOFF
        df.loc[bad, [f"{bp}_x", f"{bp}_y"]] = np.nan

    w = np.stack([df[f"{bp}_lik"].to_numpy() for bp in HEAD_PTS])
    X = np.stack([df[f"{bp}_x"].to_numpy() for bp in HEAD_PTS])
    Y = np.stack([df[f"{bp}_y"].to_numpy() for bp in HEAD_PTS])
    ok = np.isfinite(X) & np.isfinite(Y)
    w = np.where(ok, w, 0.0)
    tot = w.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        df["head_x"] = np.where(tot > 0, np.nansum(w * X, axis=0) / tot, np.nan)
        df["head_y"] = np.where(tot > 0, np.nansum(w * Y, axis=0) / tot, np.nan)
    df["head_y_up"] = IMG - 1 - df["head_y"]
    df["n_pts"] = ok.sum(axis=0)
    df["t_abs_s"] = df["abs_frame"] / FPS
    df.to_csv(KEYPOINT_CACHE, index=False)
    return df


# --------------------------------------------------------------------------- #
# cleaning
# --------------------------------------------------------------------------- #
def hampel(x, k=7, nsig=3.0):
    """Replace point outliers with the local median. Leaves NaN as NaN."""
    x = np.asarray(x, float)
    out = x.copy()
    h = k // 2
    for i in range(len(x)):
        if not np.isfinite(x[i]):
            continue
        w = x[max(0, i - h):i + h + 1]
        w = w[np.isfinite(w)]
        if len(w) < 3:
            continue
        med = np.median(w)
        mad = np.median(np.abs(w - med)) * 1.4826
        if mad > 0 and abs(x[i] - med) > nsig * mad:
            out[i] = med
    return out


def clean_series(x, max_gap=6, med_k=5):
    """Hampel -> interpolate short gaps -> median filter. Long gaps stay NaN.

    max_gap is in frames: 6 frames = 60 ms, shorter than any real ballistic
    phase here, so filling them cannot invent a movement.
    """
    x = np.asarray(x, float)
    x = hampel(x)
    good = np.isfinite(x)
    if good.sum() < 2:
        return x
    idx = np.arange(len(x))
    filled = np.interp(idx, idx[good], x[good])
    # blank runs of missing longer than max_gap
    runs = _nan_runs(good)
    for a, b in runs:
        if b - a > max_gap:
            filled[a:b] = np.nan
    sm = filled.copy()
    fin = np.isfinite(filled)
    if fin.sum() >= med_k:
        tmp = np.interp(idx, idx[fin], filled[fin])
        tmp = medfilt(tmp, med_k)
        sm = np.where(fin, tmp, np.nan)
    return sm


def _nan_runs(good):
    runs, i, n = [], 0, len(good)
    while i < n:
        if not good[i]:
            j = i
            while j < n - 1 and not good[j + 1]:
                j += 1
            runs.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    return runs


def speed(x, y, fps=FPS, smooth=7):
    """px/s from a coordinate pair; NaN wherever either input is NaN.

    Differencing position amplifies per-frame tracking jitter (1 px of jitter at
    100 fps is already 100 px/s), so the result gets a short Savitzky-Golay
    pass. The window is 70 ms -- far shorter than any jump phase here -- so it
    suppresses jitter without measurably flattening the peak.
    """
    vx, vy = np.gradient(np.asarray(x, float)), np.gradient(np.asarray(y, float))
    s = np.hypot(vx, vy) * fps
    if smooth and smooth >= 5:
        good = np.isfinite(s)
        if good.sum() > smooth:
            idx = np.arange(len(s))
            filled = np.interp(idx, idx[good], s[good])
            sm = savgol_filter(filled, smooth, 2)
            # SG overshoots either side of a sharp peak and can go negative;
            # a speed cannot be, so clamp the undershoot away
            s = np.where(good, np.maximum(sm, 0.0), np.nan)
    return s


def ang_deg(dx, dy_up):
    """Angle of a vector in the physical frame (y already flipped so up is +).
    0 deg = +x (frame-right), 90 = straight up, +-180 = frame-left."""
    return np.degrees(np.arctan2(dy_up, dx))


def wrap180(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


# --------------------------------------------------------------------------- #
# body centroid (proxy blob tracker) -- the only body-level signal available
# --------------------------------------------------------------------------- #
def load_body(cam=2):
    d = np.load(os.path.join(PROXY, f"cam{cam}_track.npz"))
    cx = d["cx"].astype(float) * PROXY_TO_FULL
    yup = d["y_up"].astype(float) * PROXY_TO_FULL
    cx, yup = medfilt(cx, 5), medfilt(yup, 5)
    return dict(cx=cx, y_up=yup, area=d["area"].astype(float),
                bbox=d["bbox"].astype(float) * PROXY_TO_FULL)


def body_at(body, frames):
    """Sample the body track at a set of absolute frames (clipped to range)."""
    f = np.clip(np.asarray(frames, int), 0, len(body["cx"]) - 1)
    return body["cx"][f], body["y_up"][f]


# --------------------------------------------------------------------------- #
# v3 keypoints -- from the onset-centred re-analysis (v3_dlc.py)
# --------------------------------------------------------------------------- #
V3_VIDEOS = os.path.join(DLC, "TreeShrewHead-sem-2026-08-27", "videos_v3")
V3_KP_CACHE = os.path.join(V3, "_keypoints_v3.csv")


def build_keypoints_v3(force=False):
    """abs_frame x cam -> 5 keypoints, from the onset-centred clips.

    These clips do not overlap each other, so unlike build_keypoints() there is
    no de-duplication step; each jump owns its own frame range.
    """
    if os.path.exists(V3_KP_CACHE) and not force:
        return pd.read_csv(V3_KP_CACHE)

    man = pd.read_csv(os.path.join(V3, "v3_clip_manifest.csv"))
    start = {(r.jump, int(r.cam)): int(r.true_start_frame) for r in man.itertuples()}

    rows = []
    for h5 in sorted(glob.glob(os.path.join(V3_VIDEOS, "*_filtered.h5"))):
        m = re.match(r"(v3_\d+)_cam(\d)", os.path.basename(h5))
        jump, cam = m.group(1), int(m.group(2))
        d = pd.read_hdf(h5)
        d.columns = d.columns.droplevel(0)
        n = len(d)
        out = pd.DataFrame({
            "abs_frame": start[(jump, cam)] + np.arange(n),
            "clip_frame": np.arange(n), "cam": cam, "jump": jump,
        })
        for bp in BODYPARTS:
            out[f"{bp}_x"] = d[(bp, "x")].to_numpy()
            out[f"{bp}_y"] = d[(bp, "y")].to_numpy()
            out[f"{bp}_lik"] = d[(bp, "likelihood")].to_numpy()
        rows.append(out)
    df = pd.concat(rows, ignore_index=True)

    # keep the pre-gate coordinates so the overlay can show what the model
    # proposed but analysis rejected (drawn hollow)
    for bp in BODYPARTS:
        df[f"{bp}_x_raw"] = df[f"{bp}_x"]
        df[f"{bp}_y_raw"] = df[f"{bp}_y"]
        df.loc[df[f"{bp}_lik"] < PCUTOFF, [f"{bp}_x", f"{bp}_y"]] = np.nan

    w = np.stack([df[f"{bp}_lik"].to_numpy() for bp in HEAD_PTS])
    X = np.stack([df[f"{bp}_x"].to_numpy() for bp in HEAD_PTS])
    Y = np.stack([df[f"{bp}_y"].to_numpy() for bp in HEAD_PTS])
    ok = np.isfinite(X) & np.isfinite(Y)
    w = np.where(ok, w, 0.0)
    tot = w.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        df["head_x"] = np.where(tot > 0, np.nansum(w * X, axis=0) / tot, np.nan)
        df["head_y"] = np.where(tot > 0, np.nansum(w * Y, axis=0) / tot, np.nan)
    df["head_y_up"] = IMG - 1 - df["head_y"]
    df["n_pts"] = ok.sum(axis=0)
    df["t_abs_s"] = df["abs_frame"] / FPS
    for bp in BODYPARTS:
        df[f"{bp}_y_up"] = IMG - 1 - df[f"{bp}_y"]
    df.to_csv(V3_KP_CACHE, index=False)
    return df


def circmean_deg(a, axis=None):
    a = np.deg2rad(np.asarray(a, float))
    s, c = np.nansum(np.sin(a), axis=axis), np.nansum(np.cos(a), axis=axis)
    n = np.sum(np.isfinite(a), axis=axis)
    out = np.degrees(np.arctan2(s, c))
    return np.where(n > 0, out, np.nan)


def circsem_deg(a, axis=None):
    """SEM of a circular mean via the resultant length (small-dispersion approx)."""
    a = np.deg2rad(np.asarray(a, float))
    n = np.sum(np.isfinite(a), axis=axis)
    s, c = np.nansum(np.sin(a), axis=axis), np.nansum(np.cos(a), axis=axis)
    R = np.sqrt(s ** 2 + c ** 2) / np.maximum(n, 1)
    R = np.clip(R, 1e-9, 1 - 1e-12)
    circstd = np.sqrt(-2 * np.log(R))              # radians
    return np.where(n > 1, np.degrees(circstd) / np.sqrt(np.maximum(n, 1)), np.nan)
