"""
Render the v4 review clips: Cam2 | Cam3 side by side with the DeepLabCut head
track drawn on top, plus a head-speed strip with a playhead.

v4 = v3 minus v3_05 / v3_07, with v3_06 and v3_12 re-timed (see v4_build.py).
Reads v4/v4_jumps.csv + v4/_keypoints_v4.csv. Window is onset - 1 s .. onset + 3 s,
4x slow, red border on the 0..0.2 s jump epoch. Output -> v4/clips/.

Drawing conventions
  filled dot  = keypoint above the 0.6 confidence gate (what analysis uses)
  hollow dot  = model reported it but below the gate -- IGNORED by analysis
  white ring  = likelihood-weighted head centroid (eye/ear/head_top/nape)
  white arrow = nape -> snout head axis, the "where the head points" vector
  red frame   = the 0..0.2 s jump epoch

  py310\\python.exe v3_overlay.py            # all jumps
  py310\\python.exe v3_overlay.py v3_02 v3_13
"""
import os
import sys
import subprocess
import numpy as np
import pandas as pd
import cv2

import v3_lib as L

V4 = os.path.join(L.AN, "v4")
OUTDIR = os.path.join(V4, "clips")
PANE = 760
STRIP = 120
SLOWMO = 4
PRE_S, POST_S = 1.0, 3.0

COLORS = {                      # BGR
    "snout":    (60, 60, 255),
    "eye":      (0, 220, 255),
    "ear":      (80, 230, 80),
    "head_top": (255, 200, 0),
    "nape":     (255, 80, 220),
}
SKELETON = [("nape", "ear"), ("ear", "eye"), ("eye", "snout"), ("nape", "head_top")]


def reader(cam, t0, dur):
    cmd = [L.FFMPEG, "-v", "error", "-nostdin", "-ss", f"{t0:.3f}", "-t", f"{dur:.3f}",
           "-i", L.SRC.format(cam), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10 ** 8)


def writer(path, w, h, fps):
    cmd = [L.FFMPEG, "-v", "error", "-nostdin", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}",
           "-i", "-", "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
           "-pix_fmt", "yuv420p", path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def draw_pane(img, row, scale):
    """img already scaled to PANE; row carries both gated and *_raw coords."""
    def P(bp, suffix=""):
        x, y = row.get(f"{bp}_x{suffix}"), row.get(f"{bp}_y{suffix}")
        if x is None or not np.isfinite(x) or not np.isfinite(y):
            return None
        return int(round(x * scale)), int(round(y * scale))

    for a, b in SKELETON:
        pa, pb = P(a), P(b)
        if pa and pb:
            cv2.line(img, pa, pb, (255, 255, 255), 1, cv2.LINE_AA)
    for bp, col in COLORS.items():
        p = P(bp)
        if p:                                   # passed the confidence gate
            cv2.circle(img, p, 5, col, -1, cv2.LINE_AA)
            cv2.circle(img, p, 5, (20, 20, 20), 1, cv2.LINE_AA)
        else:                                   # proposed but rejected
            pr = P(bp, "_raw")
            if pr:
                cv2.circle(img, pr, 5, col, 1, cv2.LINE_AA)
    hx, hy = row.get("head_x", np.nan), row.get("head_y", np.nan)
    if np.isfinite(hx) and np.isfinite(hy):
        c = (int(round(hx * scale)), int(round(hy * scale)))
        cv2.circle(img, c, 9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(img, c, (255, 255, 255), cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)
    pn, ps = P("nape", row), P("snout", row)
    if pn and ps:
        v = np.array(ps) - np.array(pn)
        n = np.linalg.norm(v)
        if n > 1:
            tip = (np.array(ps) + v / n * 42).astype(int)
            cv2.arrowedLine(img, ps, tuple(tip), (255, 255, 255), 2, cv2.LINE_AA,
                            tipLength=0.35)
    return img


def strip_img(w, h, t, speed, tnow):
    im = np.full((h, w, 3), 24, np.uint8)
    ok = np.isfinite(speed)
    if ok.sum() > 2:
        smax = max(1.0, np.nanmax(speed))
        pad = 26
        xs = ((t - t[0]) / (t[-1] - t[0]) * (w - 2 * pad) + pad).astype(int)
        ys = (h - 16 - speed / smax * (h - 34)).astype(int)
        pts = [(x, y) for x, y, g in zip(xs, ys, ok) if g]
        for i in range(1, len(pts)):
            cv2.line(im, pts[i - 1], pts[i], (120, 220, 120), 1, cv2.LINE_AA)
        x0 = int((0 - t[0]) / (t[-1] - t[0]) * (w - 2 * pad) + pad)
        cv2.line(im, (x0, 8), (x0, h - 8), (80, 80, 255), 1, cv2.LINE_AA)
        xc = int((tnow - t[0]) / (t[-1] - t[0]) * (w - 2 * pad) + pad)
        cv2.line(im, (xc, 4), (xc, h - 4), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(im, f"head speed (px/s), peak {smax:.0f}", (pad, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1, cv2.LINE_AA)
        cv2.putText(im, "onset", (x0 + 4, h - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, (80, 80, 255), 1, cv2.LINE_AA)
    return im


def render(jrow, kp):
    t0 = jrow.onset_t_s - PRE_S
    dur = PRE_S + POST_S
    f0 = int(round(t0 * L.FPS))
    nfr = int(round(dur * L.FPS))

    lut = {}
    for cam in (2, 3):
        k = kp[(kp["jump"] == jrow.jump) & (kp.cam == cam)].set_index("abs_frame")
        lut[cam] = k
    frames = np.arange(f0, f0 + nfr)
    sub = lut[2].reindex(frames)
    hx = L.clean_series(sub.head_x.to_numpy())
    hz = L.clean_series(sub.head_y_up.to_numpy())
    speed = L.speed(hx, hz)
    trel = (frames - jrow.onset_frame) / L.FPS

    W, H = PANE * 2, PANE + STRIP
    out = os.path.join(OUTDIR, f"{jrow.jump}_{jrow.cls.replace('->','_to_')}.mp4")
    rd = {cam: reader(cam, t0, dur) for cam in (2, 3)}
    wr = writer(out, W, H, L.FPS / SLOWMO)
    scale = PANE / L.IMG
    nbytes = L.IMG * L.IMG * 3

    try:
        for i, fr in enumerate(frames):
            panes = []
            for cam in (2, 3):
                buf = rd[cam].stdout.read(nbytes)
                if len(buf) < nbytes:
                    raise EOFError
                img = np.frombuffer(buf, np.uint8).reshape(L.IMG, L.IMG, 3).copy()
                img = cv2.resize(img, (PANE, PANE), interpolation=cv2.INTER_AREA)
                row = lut[cam].loc[fr].to_dict() if fr in lut[cam].index else {}
                img = draw_pane(img, row, scale)
                cv2.putText(img, f"Cam{cam}", (12, PANE - 14), cv2.FONT_HERSHEY_SIMPLEX,
                            0.62, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(img, f"Cam{cam}", (12, PANE - 14), cv2.FONT_HERSHEY_SIMPLEX,
                            0.62, (255, 255, 255), 1, cv2.LINE_AA)
                panes.append(img)
            top = np.hstack(panes)
            if 0.0 <= trel[i] <= 0.2:
                cv2.rectangle(top, (2, 2), (W - 3, PANE - 3), (60, 60, 255), 5)
            hdr = (f"{jrow.jump}  [{jrow.src_clips}]  {jrow.cls}  "
                   f"ballistic={jrow.ballistic or '?'}   {SLOWMO}x slow")
            for col, th in ((0, 0, 0), 5), ((0, 255, 255), 1):
                cv2.putText(top, hdr, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.68, col,
                            th, cv2.LINE_AA)
            sv = speed[i] if np.isfinite(speed[i]) else float("nan")
            sub2 = (f"t={trel[i]:+.2f}s  abs={fr / L.FPS:.2f}s   "
                    f"head speed={sv:.0f} px/s" if np.isfinite(sv)
                    else f"t={trel[i]:+.2f}s  abs={fr / L.FPS:.2f}s   head speed=--")
            for col, th in ((0, 0, 0), 5), ((255, 255, 255), 1):
                cv2.putText(top, sub2, (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.56, col,
                            th, cv2.LINE_AA)
            frame = np.vstack([top, strip_img(W, STRIP, trel, speed, trel[i])])
            wr.stdin.write(frame.tobytes())
    except EOFError:
        pass
    finally:
        for p in rd.values():
            p.stdout.close(); p.wait()
        wr.stdin.close(); wr.wait()
    return out


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    jumps = pd.read_csv(os.path.join(V4, "v4_jumps.csv"))
    kp = pd.read_csv(os.path.join(V4, "_keypoints_v4.csv"))
    want = sys.argv[1:]
    if want:
        jumps = jumps[jumps["jump"].isin(want)]
    for r in jumps.itertuples():
        p = render(r, kp)
        sz = os.path.getsize(p) / 1e6 if os.path.exists(p) else 0
        print(f"  {os.path.basename(p)}  {sz:.1f} MB", flush=True)
    print(f"\n-> {OUTDIR}")


if __name__ == "__main__":
    main()
