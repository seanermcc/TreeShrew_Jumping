"""
Did the cameras move during the recording?

Samples full-resolution frames at intervals and measures the global translation
of each against the first sample using phase correlation. The scene is full of
static high-contrast structure (breadboard holes, post mounts) which dominates
the correlation, so the moving animal contributes little.

If drift stays well under a pixel, the rig was rigid for the whole session and a
calibration collected now (cameras untouched since) would still be valid.
"""
import os
import subprocess
import sys

import numpy as np

FFMPEG = os.environ.get("FFMPEG_BIN", "ffmpeg")
N = 1024


def grab(path, t):
    """Decode a single full-res grayscale frame at time t seconds."""
    cmd = [FFMPEG, "-v", "error", "-nostdin", "-ss", str(t), "-i", path,
           "-frames:v", "1", "-pix_fmt", "gray", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    if len(raw) < N * N:
        return None
    return np.frombuffer(raw[: N * N], np.uint8).reshape(N, N).astype(np.float64)


def hann2d(n):
    w = np.hanning(n)
    return np.outer(w, w)


def phase_corr(a, b, win):
    """Translation of b relative to a, sub-pixel via parabolic peak fit."""
    A = np.fft.rfft2(a * win)
    B = np.fft.rfft2(b * win)
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-12
    c = np.fft.irfft2(R, a.shape)
    peak = np.unravel_index(np.argmax(c), c.shape)

    def refine(axis, idx):
        i0 = idx[axis]
        n = c.shape[axis]
        im, ip = (i0 - 1) % n, (i0 + 1) % n
        sl_m, sl_p = list(idx), list(idx)
        sl_m[axis], sl_p[axis] = im, ip
        ym, y0, yp = c[tuple(sl_m)], c[idx], c[tuple(sl_p)]
        denom = ym - 2 * y0 + yp
        delta = 0.0 if abs(denom) < 1e-12 else 0.5 * (ym - yp) / denom
        shift = i0 + delta
        if shift > n / 2:
            shift -= n
        return shift

    return refine(0, peak), refine(1, peak), float(c[peak])


def main():
    src_dir = sys.argv[1]
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 1042.0
    cams = [int(c) for c in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1, 2, 3, 4]
    times = list(np.linspace(5, duration - 5, 12))
    win = hann2d(N)

    for cam in cams:
        path = os.path.join(src_dir, f"20260730_Cam{cam}.mkv")
        if not os.path.exists(path):
            print(f"[Cam{cam}] missing, skipping")
            continue
        ref = grab(path, times[0])
        if ref is None:
            print(f"[Cam{cam}] could not decode reference frame")
            continue
        print(f"\n[Cam{cam}] translation vs t={times[0]:.0f}s")
        worst = 0.0
        for t in times[1:]:
            f = grab(path, t)
            if f is None:
                continue
            dy, dx, q = phase_corr(ref, f, win)
            mag = float(np.hypot(dy, dx))
            worst = max(worst, mag)
            print(f"   t={t:7.1f}s   dx={dx:+7.3f}  dy={dy:+7.3f}  "
                  f"|d|={mag:6.3f} px   peak={q:.4f}")
        verdict = ("RIGID" if worst < 0.5 else
                   "SLIGHT DRIFT" if worst < 2.0 else "MOVED")
        print(f"   -> max displacement {worst:.3f} px  [{verdict}]")


if __name__ == "__main__":
    main()
