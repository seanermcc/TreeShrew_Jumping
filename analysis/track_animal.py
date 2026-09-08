"""
Track the animal in a proxy movie by background subtraction.

The arena is static and the animal is the only large moving object, so a
temporal-median background plus a threshold isolates it without any training.
We keep the largest connected component per frame and record its centroid,
area and bounding box.

Coordinates are in proxy pixels (128 px == 1024 px in the original, so multiply
by 8 for full-resolution coordinates). Image y grows downward; the reported
`y_up` flips this so larger means physically higher, which is what jump
detection wants.
"""
import os
import sys

import cv2
import numpy as np

S = 128


def build_background(proxy, n_samples=600):
    idx = np.linspace(0, len(proxy) - 1, n_samples).astype(int)
    stack = np.stack([proxy[i] for i in idx]).astype(np.float32)
    return np.median(stack, axis=0)


def track(proxy, bg, min_area=8, k_mad=6.0, max_jump_px=50.0):
    """
    Picking the largest blob every frame independently flickers onto a
    same-sized static reflection whenever one is present (verified: several
    cameras show a real position alternating with a decoy ~300+ px away on
    consecutive 10 ms frames, which is kinematically impossible). Instead,
    once a track is established, prefer the candidate blob nearest the
    previous centroid; only fall back to "largest" when no prior position
    exists or nothing is within max_jump_px (proxy px, i.e. ~3x the animal's
    own size per frame -- generous even during a jump's fastest frames).
    """
    n = len(proxy)
    cx = np.full(n, np.nan, np.float32)
    cy = np.full(n, np.nan, np.float32)
    area = np.zeros(n, np.float32)
    bbox = np.zeros((n, 4), np.int16)

    # Noise scale of the static scene, so the threshold adapts to each view.
    sample = np.stack([proxy[i].astype(np.float32) - bg
                       for i in np.linspace(0, n - 1, 200).astype(int)])
    mad = np.median(np.abs(sample - np.median(sample)))
    thr = max(6.0, k_mad * mad)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    prev_xy = None
    for i in range(n):
        d = np.abs(proxy[i].astype(np.float32) - bg)
        m = (d > thr).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        nlab, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
        if nlab <= 1:
            prev_xy = None  # animal not visible; don't gate against stale position
            continue

        cand = [j for j in range(1, nlab) if stats[j, cv2.CC_STAT_AREA] >= min_area]
        if not cand:
            prev_xy = None
            continue

        if prev_xy is not None:
            d2 = [(cent[j][0] - prev_xy[0]) ** 2 + (cent[j][1] - prev_xy[1]) ** 2 for j in cand]
            k = int(np.argmin(d2))
            if d2[k] > max_jump_px ** 2:
                # nothing plausibly close -> trust largest-area instead of
                # anchoring forever to a reflection that happens to persist
                k = int(np.argmax([stats[j, cv2.CC_STAT_AREA] for j in cand]))
        else:
            k = int(np.argmax([stats[j, cv2.CC_STAT_AREA] for j in cand]))
        j = cand[k]

        area[i] = float(stats[j, cv2.CC_STAT_AREA])
        cx[i], cy[i] = cent[j]
        bbox[i] = (stats[j, cv2.CC_STAT_LEFT], stats[j, cv2.CC_STAT_TOP],
                   stats[j, cv2.CC_STAT_WIDTH], stats[j, cv2.CC_STAT_HEIGHT])
        prev_xy = cent[j]
    return cx, cy, area, bbox, thr


def main():
    proxy_dir = sys.argv[1]
    cams = [int(c) for c in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 2, 3, 4]
    for cam in cams:
        path = os.path.join(proxy_dir, f"cam{cam}_proxy.npy")
        if not os.path.exists(path):
            print(f"[Cam{cam}] no proxy, skipping")
            continue
        proxy = np.load(path, mmap_mode="r")
        print(f"[Cam{cam}] {len(proxy)} frames; building background", flush=True)
        bg = build_background(proxy)
        cx, cy, area, bbox, thr = track(proxy, bg)
        found = np.isfinite(cx)
        print(f"[Cam{cam}] threshold {thr:.2f}; animal found in "
              f"{found.mean()*100:.1f}% of frames; median area {np.median(area[found]):.0f} px")
        np.savez(os.path.join(proxy_dir, f"cam{cam}_track.npz"),
                 cx=cx, cy=cy, y_up=(S - 1 - cy), area=area, bbox=bbox,
                 background=bg, threshold=thr)


if __name__ == "__main__":
    main()
