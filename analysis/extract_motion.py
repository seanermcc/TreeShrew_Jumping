"""
Extract per-frame global signals from each camera for content-based sync.

For every frame we compute:
  mean_intensity : mean pixel value of a downscaled grayscale frame
  motion_energy  : mean |frame - prev_frame| on the downscaled frame

motion_energy is the alignment signal: all four cameras watch the same animal,
so movement transients (especially jumps) occur at the same instant in every
view even though the viewpoints differ.
"""
import os
import subprocess
import sys

import numpy as np

FFMPEG = os.environ.get("FFMPEG_BIN", "ffmpeg")
W = H = 64
FRAME_BYTES = W * H


def extract(path, progress_every=20000):
    cmd = [
        FFMPEG, "-v", "error", "-nostdin",
        "-i", path,
        "-vf", f"scale={W}:{H}:flags=fast_bilinear",
        "-pix_fmt", "gray", "-f", "rawvideo", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1 << 24)

    means, motion = [], []
    prev = None
    n = 0
    while True:
        buf = proc.stdout.read(FRAME_BYTES)
        if len(buf) < FRAME_BYTES:
            break
        cur = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        means.append(cur.mean())
        motion.append(0.0 if prev is None else np.abs(cur - prev).mean())
        prev = cur
        n += 1
        if n % progress_every == 0:
            print(f"    {n} frames", flush=True)

    proc.stdout.close()
    proc.wait()
    return np.asarray(means, np.float32), np.asarray(motion, np.float32)


def main():
    src_dir, out_dir = sys.argv[1], sys.argv[2]
    cams = [int(c) for c in sys.argv[3].split(',')] if len(sys.argv) > 3 else [1, 2, 3, 4]
    os.makedirs(out_dir, exist_ok=True)
    for cam in cams:
        path = os.path.join(src_dir, f"20260730_Cam{cam}.mkv")
        print(f"[Cam{cam}] decoding {path}", flush=True)
        means, motion = extract(path)
        np.savez(
            os.path.join(out_dir, f"cam{cam}_signals.npz"),
            mean_intensity=means, motion_energy=motion,
        )
        print(f"[Cam{cam}] {len(means)} frames -> {len(means)/100.0:.2f} s", flush=True)


if __name__ == "__main__":
    main()
