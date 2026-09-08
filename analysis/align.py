"""
Content-based synchronization check for the 4 tree-shrew cameras.

The cameras are hardware-triggered from a common Arduino pulse, so frame N of
each camera should correspond to pulse N. That guarantee breaks if (a) the
cameras were armed at different pulses (constant offset) or (b) frames were
dropped mid-recording (offset changes at the drop). The recorder writes
synthetic constant-rate timestamps, so neither shows up in the file metadata.

We recover the offset from image content: every camera watches the same animal,
so motion transients coincide across views. Cross-correlating the per-frame
motion-energy traces gives the offset directly; doing it in sliding windows
shows whether that offset is constant (case a) or steps (case b).

Lag convention (pinned by self_test, do not guess): a reported lag L for cam X
against the reference means

    camX_frame_index = ref_frame_index - L

So L = -7 means cam X is 7 frames BEHIND the reference: reference frame 0 is
cam X frame 7, and cam X's first 7 frames have no counterpart in the reference.
"""
import os
import sys

import numpy as np

EPS = 1e-9


def zscore(x):
    x = np.asarray(x, np.float64)
    return (x - x.mean()) / (x.std() + EPS)


def norm_xcorr(a, b, max_lag):
    """Normalized cross-correlation of equal-length a, b over +/- max_lag."""
    a, b = zscore(a), zscore(b)
    n = len(a)
    size = 1 << int(np.ceil(np.log2(2 * n)))
    cc = np.fft.irfft(np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size)), size)
    cc = np.concatenate((cc[-max_lag:], cc[: max_lag + 1])) / n
    return np.arange(-max_lag, max_lag + 1), cc


def best_lag(a, b, max_lag):
    lags, cc = norm_xcorr(a, b, max_lag)
    k = int(np.argmax(cc))
    return int(lags[k]), float(cc[k])


def self_test():
    """Pin down the sign convention with a known shift."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal(5000)
    shift = 7
    # b is the reference delayed by `shift`: b[i + shift] == a[i]
    a = base[shift:]
    b = base[: len(base) - shift]
    lag, corr = best_lag(a, b, 50)
    assert corr > 0.9, f"self-test correlation too low: {corr}"
    print(f"  self-test: recovered lag={lag} (expected {-shift}), corr={corr:.4f}")
    return lag


def sliding_lags(a, b, win, step, max_lag):
    """Lag estimated independently in each window -> reveals drift or steps."""
    n = min(len(a), len(b))
    out = []
    for s in range(0, n - win, step):
        wa, wb = a[s : s + win], b[s : s + win]
        if wa.std() < EPS or wb.std() < EPS:
            continue
        lag, corr = best_lag(wa, wb, max_lag)
        out.append((s, lag, corr))
    return out


def main():
    sig_dir = sys.argv[1]
    fps = 100.0
    max_lag = 500          # +/- 5 s
    win, step = 6000, 3000  # 60 s windows, 30 s hop

    print("Verifying correlation machinery")
    self_test()

    sig = {}
    for cam in (1, 2, 3, 4):
        d = np.load(os.path.join(sig_dir, f"cam{cam}_signals.npz"))
        sig[cam] = d["motion_energy"]

    print("\nFrame counts")
    counts = {c: len(sig[c]) for c in sig}
    base = counts[1]
    for c, n in counts.items():
        print(f"  Cam{c}: {n:>7} frames  {n/fps:10.2f} s   ({n-base:+d} vs Cam1)")

    n_common = min(counts.values())
    ref = 1
    print(f"\nGlobal alignment vs Cam{ref} (full recording, {n_common} frames)")
    global_lags = {}
    for c in (1, 2, 3, 4):
        lag, corr = best_lag(sig[ref][:n_common], sig[c][:n_common], max_lag)
        global_lags[c] = lag
        print(f"  Cam{ref} -> Cam{c}: lag {lag:+d} frames ({lag/fps*1000:+.0f} ms)  peak r={corr:.4f}")

    print("\nTrim needed to put all cameras on a common frame 0")
    # camX index = ref index - L. Frame k of the aligned set is camX frame k - L,
    # so a camera needs -L frames removed from its head (L>0 -> pad/see note).
    heads = {c: -global_lags[c] for c in (1, 2, 3, 4)}
    shift = min(heads.values())
    for c in (1, 2, 3, 4):
        drop = heads[c] - shift
        print(f"  Cam{c}: drop first {drop} frame(s)")

    print(f"\nStability check: lag per 60 s window (hop 30 s)")
    for c in (2, 3, 4):
        rows = sliding_lags(sig[ref][:n_common], sig[c][:n_common], win, step, 200)
        lags = np.array([r[1] for r in rows])
        corrs = np.array([r[2] for r in rows])
        good = corrs > 0.3
        print(f"\n  Cam{ref} -> Cam{c}: {len(rows)} windows, {good.sum()} with r>0.3")
        if good.sum():
            gl = lags[good]
            print(f"    lag range {gl.min():+d} .. {gl.max():+d} frames, "
                  f"median {np.median(gl):+.1f}, unique {sorted(set(gl.tolist()))[:12]}")
        for s, lag, corr in rows:
            flag = "" if corr > 0.3 else "   (low corr - ignore)"
            print(f"    t={s/fps:7.1f}s  lag {lag:+4d}  r={corr:.3f}{flag}")

    np.savez(os.path.join(sig_dir, "alignment.npz"),
             global_lags=np.array([global_lags[c] for c in (1, 2, 3, 4)]),
             counts=np.array([counts[c] for c in (1, 2, 3, 4)]))


if __name__ == "__main__":
    main()
