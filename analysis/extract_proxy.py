"""
Decode each camera once into a small uint8 "proxy movie" on disk.

Decoding the full 1024x1024 h264 takes ~5 min per camera, which makes iterating
on any analysis painful. Here we pay that cost once and write a 128x128 proxy as
a .npy that later steps memory-map, so background subtraction, blob tracking and
jump detection all run in seconds without touching the originals again.

Per-frame mean intensity and motion energy are computed in the same pass.
"""
import os
import subprocess
import sys

import numpy as np

FFMPEG = os.environ.get("FFMPEG_BIN", "ffmpeg")
S = 128
FRAME_BYTES = S * S


def frame_count(path):
    """Exact frame count from the container duration (verified CFR at 100 Hz)."""
    out = subprocess.run(
        [os.environ.get("FFPROBE_BIN", "ffprobe"), "-v", "error",
         "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True).stdout.strip()
    return int(round(float(out) * 100.0))


def run(path, out_npy, progress_every=25000):
    n_est = frame_count(path)
    proxy = np.lib.format.open_memmap(
        out_npy, mode="w+", dtype=np.uint8, shape=(n_est, S, S))

    cmd = [FFMPEG, "-v", "error", "-nostdin", "-i", path,
           "-vf", f"scale={S}:{S}:flags=area",
           "-pix_fmt", "gray", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1 << 24)

    means = np.zeros(n_est, np.float32)
    motion = np.zeros(n_est, np.float32)
    prev = None
    i = 0
    while i < n_est:
        buf = proc.stdout.read(FRAME_BYTES)
        if len(buf) < FRAME_BYTES:
            break
        cur = np.frombuffer(buf, np.uint8).reshape(S, S)
        proxy[i] = cur
        curf = cur.astype(np.float32)
        means[i] = curf.mean()
        motion[i] = 0.0 if prev is None else np.abs(curf - prev).mean()
        prev = curf
        i += 1
        if i % progress_every == 0:
            print(f"    {i}/{n_est}", flush=True)

    proc.stdout.close()
    proc.wait()
    proxy.flush()

    if i != n_est:
        print(f"    NOTE: decoded {i} frames, container implied {n_est}")
    return i, means[:i], motion[:i]


def main():
    src_dir, out_dir = sys.argv[1], sys.argv[2]
    cams = [int(c) for c in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1, 2, 3, 4]
    os.makedirs(out_dir, exist_ok=True)
    for cam in cams:
        path = os.path.join(src_dir, f"20260730_Cam{cam}.mkv")
        print(f"[Cam{cam}] decoding -> proxy", flush=True)
        n, means, motion = run(path, os.path.join(out_dir, f"cam{cam}_proxy.npy"))
        np.savez(os.path.join(out_dir, f"cam{cam}_signals.npz"),
                 mean_intensity=means, motion_energy=motion, n_frames=n)
        print(f"[Cam{cam}] {n} frames = {n/100.0:.2f} s", flush=True)


if __name__ == "__main__":
    main()
