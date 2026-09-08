"""
Build the v3 jump set: apply the manual confirmation, merge split events, define
jump onset, and render DLC-overlay review videos.

Input  : jump_clips_v2/classification_v2.csv  (manual confirmed_jump / ballistic)
Output : v3/v3_jumps.csv          one row per accepted jump
         v3/clips/v3_NN_*.mp4     Cam2|Cam3 with the DLC head track drawn on

Decisions encoded here, from the manual pass:
  - drop every clip with confirmed_jump == 'no'
  - keep the single 'maybe' (v2_01), flagged so it can be excluded later
  - merge v2_20+v2_21 and v2_25+v2_26 -- each pair is one jump the v2 bout
    splitter cut in half ("jump cut at end" / "combine w/ 25 for full jump")

Jump onset (t = 0 everywhere downstream) is the frame of PEAK Cam2 head speed
inside the event window. It is required to sit on real tracked frames, not on
interpolated ones, so a dropout cannot masquerade as the fastest moment.
"""
import os
import subprocess
import numpy as np
import pandas as pd

import v3_lib as L

V2 = os.path.join(L.AN, "jump_clips_v2")
CLASS_CSV = os.path.join(V2, "classification_v2.csv")
OUT = L.V3
CLIPDIR = os.path.join(OUT, "clips")

PRE_S, POST_S = 1.0, 3.0        # analysis / clip window around onset
MERGES = [("v2_20", "v2_21"), ("v2_25", "v2_26")]

# --- arena geometry, Cam2, measured off static frames and validated against
# --- both the jump trajectories and the forepaw-touchdown annotations
# --- (see fig05_geometry_check.png / fig14_paw_annotation_check.png)
GEOM = dict(
    plat_surface_y_up=675.0,     # both platforms, top surface (feet sit here)
    left_edge_x=350.0,           # left platform, front (gap-facing) edge
    right_edge_x=592.0,          # right platform, front (gap-facing) edge
    floor_y_up=210.0,            # breadboard floor in the drop-landing zone.
                                 # NB the floor is not level in-image (perspective):
                                 # ~305 further back, ~210 where the down-jumps land.
)


def load_events():
    # saved from Excel -> cp1252 (curly apostrophe in one of the notes)
    c = pd.read_csv(CLASS_CSV, encoding="cp1252")
    c["confirmed_jump"] = c["confirmed_jump"].astype(str).str.strip().str.lower()
    c["ballistic"] = c["ballistic"].astype(str).str.strip().str.lower()
    # NB: `c.clip` is the DataFrame.clip METHOD -- always index with c["clip"]
    # Group FIRST, then filter: a merge partner may itself be marked 'no'
    # ("jump cut at end") while the pair together is one real jump.
    group = {}
    for a, b in MERGES:
        group[a] = group[b] = (a, b)

    KEEP = ("y", "yes", "maybe")
    rows, seen = [], set()
    for r in c.itertuples():
        key = group.get(getattr(r, "clip"), (getattr(r, "clip"),))
        if key in seen:
            continue
        seen.add(key)
        parts = [c[c["clip"] == k].iloc[0] for k in key if len(c[c["clip"] == k])]
        if not any(str(p.confirmed_jump) in KEEP for p in parts):
            continue
        # the class / ballistic call belongs to the half the user confirmed
        carrier = next((p for p in parts if str(p.confirmed_jump) in ("y", "yes")),
                       parts[0])
        notes = " | ".join(str(p.notes) for p in parts
                           if str(p.notes) not in ("nan", ""))
        rows.append(dict(
            src_clips="+".join(key), orig="+".join(str(p.orig_detections) for p in parts),
            cls=carrier.class_updown, confirmed=carrier.confirmed_jump,
            ballistic=str(carrier.ballistic),
            note=notes,
            t0=min(p.t_takeoff_s for p in parts),
            t1=max(p.t_land_s for p in parts)))
    ev = pd.DataFrame(rows).sort_values("t0").reset_index(drop=True)
    ev["ballistic"] = ev["ballistic"].replace({"nan": "", "none": ""})
    return ev


def find_onset(kp2, f0, f1):
    """Frame of peak Cam2 head speed in [f0, f1], restricted to frames whose
    speed is computed from genuinely tracked (not gap-filled) samples."""
    w = kp2[(kp2.abs_frame >= f0 - 30) & (kp2.abs_frame <= f1 + 30)].sort_values("abs_frame")
    if len(w) < 5:
        return None
    fr = w.abs_frame.to_numpy()
    x = L.clean_series(w.head_x.to_numpy())
    z = L.clean_series(w.head_y_up.to_numpy())
    # smooth=0 on purpose: onset is DEFINED as the frame of peak raw head speed,
    # and the v3 clips are cut around it. Smoothing here moves the global max by
    # up to 2.5 s on jumps that have two comparable peaks, which would desync
    # the cut clips. Display/analysis smoothing is applied separately.
    sp = L.speed(x, z, smooth=0)
    real = np.isfinite(w.head_x.to_numpy())
    # a peak is only trusted if it and both neighbours were really tracked
    solid = real.copy()
    solid[1:-1] &= real[:-2] & real[2:]
    inwin = (fr >= f0) & (fr <= f1)
    cand = inwin & solid & np.isfinite(sp)
    if not cand.any():
        cand = inwin & np.isfinite(sp)
        if not cand.any():
            return None
    i = np.nanargmax(np.where(cand, sp, -np.inf))
    return int(fr[i]), float(sp[i]), bool(solid[i])


def main():
    os.makedirs(CLIPDIR, exist_ok=True)
    kp = L.build_keypoints()
    kp2 = kp[kp.cam == 2]

    ev = load_events()
    out = []
    for i, r in enumerate(ev.itertuples(), 1):
        f0, f1 = int(r.t0 * L.FPS), int(r.t1 * L.FPS)
        got = find_onset(kp2, f0, f1)
        if got is None:
            print(f"  !! no onset for {r.src_clips}")
            continue
        onset_f, pk, solid = got
        out.append(dict(
            jump=f"v3_{i:02d}", src_clips=r.src_clips, orig_detections=r.orig,
            cls=r.cls, ballistic=r.ballistic, confirmed=r.confirmed,
            onset_frame=onset_f, onset_t_s=round(onset_f / L.FPS, 3),
            peak_head_speed_pxs=round(pk), onset_on_tracked_frames=solid,
            event_t0_s=r.t0, event_t1_s=r.t1,
            win_t0_s=round(onset_f / L.FPS - PRE_S, 3),
            win_t1_s=round(onset_f / L.FPS + POST_S, 3),
            note=r.note))
    df = pd.DataFrame(out)

    # how much of the -1..+3 s window has real Cam2 / Cam3 keypoint coverage?
    for cam in (2, 3):
        k = kp[kp.cam == cam]
        have = set(k.abs_frame[np.isfinite(k.head_x)].tolist())
        cov = []
        for r in df.itertuples():
            fr = np.arange(int(r.win_t0_s * L.FPS), int(r.win_t1_s * L.FPS))
            cov.append(round(100 * np.mean([f in have for f in fr])))
        df[f"cam{cam}_cov_pct"] = cov

    df.to_csv(os.path.join(OUT, "v3_jumps.csv"), index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(df.to_string(index=False))
    print(f"\n{len(df)} jumps -> {os.path.join(OUT, 'v3_jumps.csv')}")
    print("class:", df.cls.value_counts().to_dict())
    print("ballistic:", df.ballistic.value_counts(dropna=False).to_dict())
    return df


if __name__ == "__main__":
    main()
