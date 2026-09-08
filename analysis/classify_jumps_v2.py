"""
jump_clips_v2 : find the *actual* jump events and cut tight review clips.

The 27 clips in analysis/jump_clips are raw motion-energy triggers: several fire
around one real relocation, some fire on non-jumps. This script works from the
DeepLabCut head track (analysis/dlc/head_pose/_all_headpose.csv), which is clean
where the blob tracker is not, to:

  1. stitch the overlapping 6 s DLC windows into continuous per-segment traces
     (Cam2 = clean side profile, primary; Cam3 = oblique, cross-check)
  2. de-glitch the head centroid and derive head speed
  3. split each segment into movement BOUTS separated by sustained stillness,
     merging bouts <1.5 s apart -> one bout per real relocation
  4. for each bout: takeoff/landing time, resting head height just before and
     just after (-> platform vs ground), peak head speed, displacement
  5. classify  up->up | up->down | down->up  from the pre/post heights
  6. write jump_clips_v2/classification.csv and cut a Cam2|Cam3 side-by-side
     clip per event: 1 s before takeoff to 1 s after landing, 4x slow-mo.

'ballistic' (true jump vs walk/climb/crawl) is left blank for manual scoring,
as is 'confirmed_jump'.

Run:  analysis\dlc\py310\python.exe analysis\classify_jumps_v2.py
"""
import os
import sys
import subprocess
import numpy as np
import pandas as pd
from scipy.signal import medfilt

FPS = 100.0
AN = os.path.dirname(os.path.abspath(__file__))
HEADPOSE = os.path.join(AN, "dlc", "head_pose", "_all_headpose.csv")
SRC = os.path.join(os.path.dirname(AN), "JumpingTest_v6_07302026", "20260730_Cam{}.mkv")
OUTDIR = os.path.join(AN, "jump_clips_v2")
FFMPEG = os.path.join(AN, "dlc", "py310", "Lib", "site-packages",
                      "imageio_ffmpeg", "binaries", "ffmpeg-win-x86_64-v7.1.exe")
FONT = r"C\:/Windows/Fonts/arial.ttf"

# ---- tunables (px are full-res 1024; from resting-height histograms + video) ----
PLAT_C2 = 620.0        # Cam2 head_y_up >= this  => head is at platform height
PLAT_C3 = 690.0        # Cam3 head_y_up >= this
AMBIG_C2 = 45.0        # if |z_pre-PLAT_C2| < this, defer to Cam3
MERGE_GAP_S = 1.0      # merge bouts within a segment closer than this ...
MERGE_MAX_SPAN_S = 5.0 # ... but never merge into a span longer than this
KEEP_NET = 120.0       # keep bout as event if pre->post head move >= this (px)
KEEP_DZ = 85.0         # ...or vertical change >= this
KEEP_PEAK = 2100.0     # ...or peak head speed >= this (px/s)
KEEP_DET_NET = 70.0    # ...or it covers an original detection and moved >= this
REST_LOOK_S = 3.0      # search this far for a settled window each side
REST_MIN_S = 0.22      # a settled window must be at least this long
PAD_S = 1.0            # clip pad before takeoff / after landing
CLIP_MAX_S = 7.0
SLOWMO = 4

DET = {1:352.97,2:390.19,3:453.32,4:457.14,5:463.89,6:475.23,7:481.7,8:580.34,
       9:583.22,10:591.88,11:608.04,12:613.74,13:643.25,14:649.59,15:651.1,
       16:661.84,17:672.76,18:678.69,19:682.55,20:684.06,21:688.83,22:827.53,
       23:831.19,24:856.84,25:894.03,26:911.9,27:935.45}


# --------------------------------------------------------------------------- #
def hampel(x, k=7, nsig=3.0):
    x = x.astype(float).copy()
    out = x.copy()
    h = k // 2
    for i in range(len(x)):
        w = x[max(0, i - h):i + h + 1]
        w = w[np.isfinite(w)]
        if len(w) < 3:
            continue
        med = np.median(w)
        mad = np.median(np.abs(w - med)) * 1.4826
        if mad > 0 and abs(x[i] - med) > nsig * mad:
            out[i] = med
    return out


def load_segments(df):
    wins = df.groupby("jump_no")["abs_frame"].agg(["min", "max"]).sort_values("min")
    segs, cur = [], [wins.index[0]]
    for j in wins.index[1:]:
        if wins.loc[j, "min"] <= wins.loc[cur[-1], "max"] + 1:
            cur.append(j)
        else:
            segs.append(cur); cur = [j]
    segs.append(cur)
    return segs


def stitch(df, seg, cam):
    s = df[(df.jump_no.isin(seg)) & (df.cam == cam)].copy()
    s["central"] = (s["clip_frame"] - 300).abs()
    s = (s.sort_values(["abs_frame", "central"])
           .drop_duplicates("abs_frame", keep="first")
           .sort_values("abs_frame").reset_index(drop=True))
    good = np.isfinite(s["head_x"]) & np.isfinite(s["head_y_up"])
    sg = s[good]
    f = sg["abs_frame"].to_numpy()
    full = np.arange(s["abs_frame"].min(), s["abs_frame"].max() + 1)
    present = np.isin(full, f)
    nearest = np.abs(full[:, None] - f[None, :]).min(axis=1)
    x = hampel(np.interp(full, f, sg["head_x"].to_numpy()))
    z = hampel(np.interp(full, f, sg["head_y_up"].to_numpy()))
    xs, zs = medfilt(x, 5), medfilt(z, 5)
    speed = np.hypot(np.gradient(xs), np.gradient(zs)) * FPS
    speed[nearest > 4] = np.nan
    return dict(frame=full, t=full / FPS, x=xs, z=zs, speed=speed,
               present=present, gaplen=nearest)


def bouts(tr, move_on=650.0, move_off=200.0, still_speed=130.0,
          still_run_s=0.30, min_move_frames=4, min_peak=650.0):
    sp = tr["speed"]
    sp_f = np.where(np.isfinite(sp), sp, 0.0)
    n = len(sp_f)
    biggap = tr["gaplen"] > 4
    still = (sp_f < still_speed) & (tr["gaplen"] <= 2)
    sr = int(still_run_s * FPS)
    rest = np.zeros(n, bool)
    i = 0
    while i < n:
        if still[i]:
            j = i
            while j < n - 1 and still[j + 1]:
                j += 1
            if j - i + 1 >= sr:
                rest[i:j + 1] = True
            i = j + 1
        else:
            i += 1
    notrest = ~rest
    spans, i = [], 0
    while i < n:
        if notrest[i]:
            j = i
            while j < n - 1 and notrest[j + 1]:
                j += 1
            spans.append((i, j)); i = j + 1
        else:
            i += 1
    out = []
    for a, b in spans:
        seg_sp = sp_f[a:b + 1]
        has_gap = biggap[a:b + 1].any()
        pk = np.nanmax(np.where(np.isfinite(sp[a:b + 1]), sp[a:b + 1], 0))
        if not ((seg_sp > move_on).any() or has_gap):
            continue
        if pk < min_peak and not has_gap:
            continue
        idx = np.where((seg_sp > move_off) | biggap[a:b + 1])[0]
        if len(idx) < min_move_frames and not has_gap:
            continue
        out.append([a + idx[0], a + idx[-1]])
    # merge adjacent bouts only if close AND the merged span stays short
    merged = []
    for r in out:
        if (merged
                and (tr["t"][r[0]] - tr["t"][merged[-1][1]]) < MERGE_GAP_S
                and (tr["t"][r[1]] - tr["t"][merged[-1][0]]) <= MERGE_MAX_SPAN_S):
            merged[-1][1] = r[1]
        else:
            merged.append(list(r))
    return merged


def rest_pos(tr, i, direction):
    """Earliest sustained (>=REST_MIN_S) settled window within REST_LOOK_S of i,
    walking in `direction`. Returns (x_med, z_med, n_frames_used)."""
    n = len(tr["t"])
    step = 1 if direction > 0 else -1
    look = int(REST_LOOK_S * FPS)
    need = int(REST_MIN_S * FPS)
    run = []
    best = None
    for k in range(1, look):
        j = i + step * k
        if j < 0 or j >= n:
            break
        s = tr["speed"][j]
        calm = np.isfinite(s) and s < 220 and tr["gaplen"][j] <= 3
        if calm:
            run.append(j)
            if len(run) >= need:
                best = list(run)          # keep growing; take the whole rest
        else:
            if best:
                break
            run = []
    if not best:
        j = max(0, min(n - 1, i))
        w = range(max(0, j - 3), min(n, j + 4))
        return float(np.median(tr["x"][list(w)])), float(np.median(tr["z"][list(w)])), 0
    a = np.array(best)
    return float(np.median(tr["x"][a])), float(np.median(tr["z"][a])), len(a)


def classify(z2p, z2q, z3p, z3q):
    def side(z2, z3):
        if abs(z2 - PLAT_C2) < AMBIG_C2 and np.isfinite(z3):
            return z3 >= PLAT_C3
        return z2 >= PLAT_C2
    p, q = side(z2p, z3p), side(z2q, z3q)
    return {(1, 1): "up->up", (1, 0): "up->down",
            (0, 1): "down->up", (0, 0): "ground->ground"}[(int(p), int(q))]


# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    df = pd.read_csv(HEADPOSE)
    segs = load_segments(df)

    events = []
    for si, seg in enumerate(segs):
        c2 = stitch(df, seg, 2)
        c3 = stitch(df, seg, 3)
        for i0, i1 in bouts(c2):
            t0, t1 = float(c2["t"][i0]), float(c2["t"][i1])
            x2p, z2p, npre = rest_pos(c2, i0, -1)
            x2q, z2q, npost = rest_pos(c2, i1, +1)

            def c3z(fr, direction):
                k = int(np.clip(np.searchsorted(c3["frame"], fr), 0, len(c3["frame"]) - 1))
                w = c3["z"][max(0, k - 20):k + 1] if direction < 0 else c3["z"][k:k + 20]
                w = w[np.isfinite(w)]
                return float(np.median(w)) if len(w) else np.nan
            z3p = c3z(c2["frame"][i0] - 5, -1)
            z3q = c3z(c2["frame"][i1] + 5, +1)

            seg_sp = c2["speed"][i0:i1 + 1]
            pk = float(np.nanmax(seg_sp)) if np.isfinite(seg_sp).any() else np.nan
            # glitch check: 1-frame reversals in x
            xw = c2["x"][i0:i1 + 1]
            dv = np.diff(xw)
            glitchy = np.any((np.abs(dv[:-1]) > 55) & (np.sign(dv[:-1]) != np.sign(dv[1:]))
                             & (np.abs(dv[1:]) > 55))
            gapfrac = float(np.mean(c2["gaplen"][i0:i1 + 1] > 4))
            dz = z2q - z2p
            dx = x2q - x2p
            net = float(np.hypot(dx, dz))
            srcs = sorted(j for j in seg if t0 - 0.8 <= DET[j] <= t1 + 0.8)
            keep = ((net >= KEEP_NET) or (abs(dz) >= KEEP_DZ) or (pk >= KEEP_PEAK)
                    or (srcs and net >= KEEP_DET_NET and (t1 - t0) >= 0.5))

            cls = classify(z2p, z2q, z3p, z3q)
            qual = "glitch" if glitchy else ("gappy" if gapfrac > 0.25 else "ok")
            events.append(dict(
                seg=si, t0=t0, t1=t1, srcs=srcs, keep=keep, cls=cls,
                dur_ms=round((t1 - t0) * 1000), peak=round(pk) if np.isfinite(pk) else -1,
                dz=round(dz), net=round(net),
                z2_pre=round(z2p), z2_post=round(z2q),
                z3_pre=round(z3p) if np.isfinite(z3p) else "",
                z3_post=round(z3q) if np.isfinite(z3q) else "",
                npre=npre, npost=npost, qual=qual, gapfrac=round(gapfrac, 2)))

    kept = [e for e in events if e["keep"] and e["cls"] != "ground->ground"]
    dropped = [e for e in events if not (e["keep"] and e["cls"] != "ground->ground")]

    rows = []
    for n, e in enumerate(kept, 1):
        ck_t0 = max(0.0, e["t0"] - PAD_S)
        ck_t1 = min(e["t1"] + PAD_S, ck_t0 + CLIP_MAX_S)  # keep takeoff, trim tail
        # first-pass assessment (the user overrides via confirmed_jump / ballistic)
        if e["cls"] == "up->down":
            a = "platform exit - jump/scramble down"
        elif e["cls"] == "down->up":
            a = "platform entry - climb/jump up"
        elif e["net"] >= 240 or e["peak"] >= 2200:
            a = "on-platform traverse"
        else:
            a = "small reposition - may not be a jump"
        if not e["srcs"] and (e["t1"] - e["t0"]) < 1.5 and e["net"] < 200:
            a = "minor movement - likely not a distinct jump"
        if e["qual"] == "glitch":
            a += " [peak speed may be a tracking glitch]"
        elif e["qual"] == "gappy":
            a += " [head track patchy through the fast part]"
        rows.append(dict(
            clip=f"v2_{n:02d}",
            orig_detections="+".join(f"{j:02d}" for j in e["srcs"]) or "(none)",
            class_updown=e["cls"],
            confirmed_jump="", ballistic="",
            t_takeoff_s=round(e["t0"], 2), t_land_s=round(e["t1"], 2),
            dur_ms=e["dur_ms"], peak_head_speed_pxs=e["peak"],
            dz_px=e["dz"], net_disp_px=e["net"],
            cam2_z_pre=e["z2_pre"], cam2_z_post=e["z2_post"],
            cam3_z_pre=e["z3_pre"], cam3_z_post=e["z3_post"],
            track_quality=e["qual"], gap_frac=e["gapfrac"],
            clip_start_s=round(ck_t0, 2), clip_end_s=round(ck_t1, 2),
            claude_assessment=a, notes=""))
    out = pd.DataFrame(rows)
    csv_path = os.path.join(OUTDIR, "classification.csv")
    out.to_csv(csv_path, index=False)

    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(out.to_string(index=False))
    print(f"\n{len(kept)} events kept -> {csv_path}")
    print(f"{len(dropped)} sub-threshold/again dropped:")
    for e in dropped:
        print(f"  seg{e['seg']:>2} {e['t0']:7.2f}-{e['t1']:7.2f}  {e['cls']:<14} "
              f"dur={e['dur_ms']:>4} peak={e['peak']:>5} net={e['net']:>4} "
              f"dz={e['dz']:>4} srcs={e['srcs']}")
    print(f"\nclass counts: {out.class_updown.value_counts().to_dict()}")

    # ---- cut clips ----
    if "--nocut" not in sys.argv:
        for r in rows:
            cut_clip(r)
    print(f"\ndone -> {OUTDIR}")


def cut_clip(r):
    t0, t1 = r["clip_start_s"], r["clip_end_s"]
    dur = t1 - t0
    out_path = os.path.join(OUTDIR, f"{r['clip']}.mp4")
    ins = []
    for cam in (2, 3):
        ins += ["-ss", f"{t0:.3f}", "-t", f"{dur:.3f}", "-i", SRC.format(cam)]
    pane = 760
    take_rel = r["t_takeoff_s"] - t0
    land_rel = r["t_land_s"] - t0
    lab = (f"v2_{r['clip'][3:]}   det {r['orig_detections']}   "
           f"{r['class_updown']}   {SLOWMO}x slow")
    tsec = f"{t0}+t/{SLOWMO}"
    filt = (
        f"[0:v]scale={pane}:{pane},setpts={SLOWMO}*PTS,"
        f"drawtext=fontfile='{FONT}':text='Cam2  (side)':x=12:y=50:fontsize=24:"
        f"fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=5[a];"
        f"[1:v]scale={pane}:{pane},setpts={SLOWMO}*PTS,"
        f"drawtext=fontfile='{FONT}':text='Cam3  (oblique)':x=w-text_w-12:y=50:fontsize=24:"
        f"fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=5[b];"
        f"[a][b]hstack=inputs=2[s];"
        f"[s]drawtext=fontfile='{FONT}':text='{lab}':x=(w-text_w)/2:y=6:fontsize=26:"
        f"fontcolor=yellow:box=1:boxcolor=black@0.65:boxborderw=6,"
        f"drawtext=fontfile='{FONT}':"
        f"text='t=%{{eif\\:{tsec}\\:d}}.%{{eif\\:mod(({tsec})*100\\,100)\\:d\\:2}}s':"
        f"x=(w-text_w)/2:y=h-40:fontsize=26:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=5,"
        f"drawbox=x=0:y=0:w=iw:h=ih:color=red@0.9:t=8:"
        f"enable='between(t\\,{take_rel*SLOWMO}\\,{land_rel*SLOWMO})'[out]"
    )
    cmd = [FFMPEG, "-v", "error", "-nostdin", *ins, "-filter_complex", filt,
           "-map", "[out]", "-r", f"{FPS/SLOWMO:.3f}", "-c:v", "libx264",
           "-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-y", out_path]
    p = subprocess.run(cmd, capture_output=True, text=True)
    sz = os.path.getsize(out_path) / 1e6 if os.path.exists(out_path) else 0
    print(f"  {r['clip']}  [{t0:6.2f},{t1:6.2f}]  {r['class_updown']:<14} "
          f"{'OK' if p.returncode == 0 else 'FAIL'}  {sz:.1f} MB"
          + ("" if p.returncode == 0 else f"\n{p.stderr[-1500:]}"))


if __name__ == "__main__":
    main()
