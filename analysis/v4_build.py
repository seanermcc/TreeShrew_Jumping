"""
v4 jump set = v3 with manual corrections from the user's frame-by-frame review:

  * DROP  v3_05 and v3_07  -- not jumps ("crawls across" the platform)
  * DROP  v3_12  -- a fall, not a real jump ("crawls + falls a bit"). Was briefly
           kept (re-timed to f68255) in the first v4 pass; the user's second
           review (2026-09-08) reclassified it as a fall and pulled it out of
           every downstream analysis and out of v4/clips.
  * RETIME v3_06  -> onset f48158 (t = -0.32 s vs the v3 onset): the up->down drop
           off the platform. The v3 peak-speed detector rejected the true peak
           (keypoints drop out in the motion blur of the drop) and settled on a
           smaller local max 0.32 s later.

Jump IDs are UNCHANGED (still v3_01..v3_17, with 05 / 07 / 12 absent) so every
existing v3 figure / CSV still lines up.

Keypoints: the 12 untouched jumps reuse v3's onset-centred re-analysis
(_keypoints_v3.csv). v3_06 is re-sliced from the master keypoint table
(the original overlapping DLC windows, _keypoints_master.csv) because its
analysis window shifted outside the v3 re-cut clip. Coverage is comparable
(master vs v3 re-cut agree to ~0.12 px median, per v3/README.md).

Output: v4/v4_jumps.csv, v4/_keypoints_v4.csv
"""
import os
import numpy as np
import pandas as pd

import v3_lib as L

V4 = os.path.join(L.AN, "v4")
os.makedirs(V4, exist_ok=True)

DROP = ["v3_05", "v3_07", "v3_12"]
ONSET_OVERRIDE = {
    "v3_06": (48158, "manual: movement onset, t-0.32s vs v3 (platform drop)"),
}
PRE_S, POST_S = 1.0, 3.0
RAW_COLS = ([f"{bp}_x_raw" for bp in L.BODYPARTS]
           + [f"{bp}_y_raw" for bp in L.BODYPARTS]
           + [f"{bp}_y_up" for bp in L.BODYPARTS])


def speed_at(kpc, frame):
    """Cleaned Cam2 head speed (px/s) at a single abs frame, windowed context."""
    w = kpc[(kpc.abs_frame >= frame - 40) & (kpc.abs_frame <= frame + 40)] \
        .sort_values("abs_frame")
    if len(w) < 6:
        return np.nan
    fr = w.abs_frame.to_numpy()
    sp = L.speed(L.clean_series(w.head_x.to_numpy()),
                 L.clean_series(w.head_y_up.to_numpy()), smooth=0)
    i = np.where(fr == frame)[0]
    return float(sp[i[0]]) if len(i) and np.isfinite(sp[i[0]]) else np.nan


def main():
    v3 = pd.read_csv(os.path.join(L.V3, "v3_jumps.csv"))
    v3 = v3[~v3.jump.isin(DROP)].reset_index(drop=True)

    master = L.build_keypoints()
    kp_v3 = L.build_keypoints_v3()

    kp_rows, out = [], []
    for r in v3.itertuples():
        j = r.jump
        if j in ONSET_OVERRIDE:
            onset, src = ONSET_OVERRIDE[j]
            # slice the master table around the new onset, tag it like kp_v3
            sub = master[(master.abs_frame >= onset - 180)
                         & (master.abs_frame <= onset + 380)].copy()
            sub.insert(3, "jump", j)
            for c in RAW_COLS:                       # schema parity with kp_v3
                if c.endswith("_y_up"):
                    bp = c[:-5]
                    sub[c] = L.IMG - 1 - sub[f"{bp}_y"]
                else:                                # no pre-gate coords kept in
                    sub[c] = sub[c.replace("_raw", "")]   # master -> reuse gated
            kp_rows.append(sub)
        else:
            onset, src = int(r.onset_frame), "v3 peak head speed (unchanged)"
            kp_rows.append(kp_v3[kp_v3.jump == j].copy())

        kpc = master[(master.cam == 2)]
        pk = speed_at(kpc, onset)
        win_t0, win_t1 = onset / L.FPS - PRE_S, onset / L.FPS + POST_S
        rec = dict(
            jump=j, src_clips=r.src_clips, orig_detections=r.orig_detections,
            cls=r.cls, ballistic=r.ballistic, confirmed=r.confirmed,
            onset_frame=onset, onset_t_s=round(onset / L.FPS, 3),
            onset_source=src,
            peak_head_speed_pxs=round(pk) if np.isfinite(pk) else "",
            v3_onset_frame=int(r.onset_frame),
            d_onset_s=round(onset / L.FPS - r.onset_frame / L.FPS, 3),
            event_t0_s=r.event_t0_s, event_t1_s=r.event_t1_s,
            win_t0_s=round(win_t0, 3), win_t1_s=round(win_t1, 3),
            note=r.note)
        out.append(rec)

    kp = pd.concat(kp_rows, ignore_index=True)
    kp.to_csv(os.path.join(V4, "_keypoints_v4.csv"), index=False)

    df = pd.DataFrame(out)
    for cam in (2, 3):
        k = master[master.cam == cam]
        have = set(k.abs_frame[np.isfinite(k.head_x)].tolist())
        cov = []
        for r in df.itertuples():
            fr = np.arange(int(r.win_t0_s * L.FPS), int(r.win_t1_s * L.FPS))
            cov.append(round(100 * np.mean([f in have for f in fr])))
        df[f"cam{cam}_cov_pct"] = cov

    df.to_csv(os.path.join(V4, "v4_jumps.csv"), index=False)
    pd.set_option("display.width", 260, "display.max_columns", 40)
    print(df[["jump", "cls", "ballistic", "onset_frame", "onset_t_s",
              "d_onset_s", "peak_head_speed_pxs", "cam2_cov_pct",
              "cam3_cov_pct", "onset_source"]].to_string(index=False))
    print(f"\n{len(df)} jumps -> {os.path.join(V4, 'v4_jumps.csv')}")
    print(f"keypoints: {len(kp)} rows -> {os.path.join(V4, '_keypoints_v4.csv')}")


if __name__ == "__main__":
    main()
