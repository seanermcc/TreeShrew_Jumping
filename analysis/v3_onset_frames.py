"""
Pre-onset frame montage.

For every v3 jump, grab the Cam2 frame at -0.10 s, -0.05 s and 0 s relative to
jump onset (onset = the frame of peak Cam2 head-centroid speed, see v3_build.py).
Draw the DLC head keypoints on, crop around the head, and lay the jumps out
8 per figure, 2 columns x 4 rows (each cell = the 3-frame strip for one jump).

    E:\TreeShrew_Jumping\analysis\dlc\py310\python.exe onset_triplets.py [cam]

Outputs (in analysis/v3/):
    onset_frames/<jump>_cam<c>_{m100,m050,000}.png   individual crops (overlay)
    onset_frames/<jump>_cam<c>_{...}_clean.png        individual crops (no overlay)
    onset_frames/index.csv
    figures/fig_onset_triplets_cam<c>_g{1,2,3}.png    the montages
"""
import os
import sys
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, r"E:\TreeShrew_Jumping\analysis")
import v3_lib as L

CAM = int(sys.argv[1]) if len(sys.argv) > 1 else 2
OFFSETS_S = [-0.10, -0.05, 0.0]
TAGS = ["m100", "m050", "000"]
LBL = ["-0.10 s", "-0.05 s", "onset (0 s)"]
CROP = 520                      # square crop side, px (source is 1024x1024)
PER_FIG = 8                     # 2 cols x 4 rows

OUTDIR = os.path.join(L.V3, "onset_frames")
FIGDIR = os.path.join(L.V3, "figures")
os.makedirs(OUTDIR, exist_ok=True)

KP_COLORS = [("snout", "#ff3b3b"), ("eye", "#ffd400"), ("ear", "#39d353"),
             ("head_top", "#00c8ff"), ("nape", "#ff5cdc")]


def grab_frame(cam, frame_idx):
    """Exact Cam<cam> frame `frame_idx` (100 fps source) as a gray 1024x1024 array."""
    t = frame_idx / L.FPS
    cmd = [L.FFMPEG, "-v", "error", "-nostdin", "-ss", f"{t:.4f}",
           "-i", L.SRC.format(cam), "-frames:v", "1", "-f", "rawvideo",
           "-pix_fmt", "gray", "-"]
    p = subprocess.run(cmd, capture_output=True)
    buf = p.stdout
    if len(buf) < L.IMG * L.IMG:
        return np.zeros((L.IMG, L.IMG), np.uint8)
    return np.frombuffer(buf[:L.IMG * L.IMG], np.uint8).reshape(L.IMG, L.IMG)


def crop_box(cx, cy):
    x0 = int(round(np.clip(cx - CROP / 2, 0, L.IMG - CROP)))
    y0 = int(round(np.clip(cy - CROP / 2, 0, L.IMG - CROP)))
    return x0, y0


def main():
    jumps = pd.read_csv(os.path.join(L.V3, "v3_jumps.csv"))
    jumps["ballistic"] = jumps["ballistic"].fillna("").astype(str)
    kp = L.build_keypoints_v3()
    kc = kp[kp.cam == CAM].set_index(["jump", "abs_frame"])
    try:
        body = L.load_body(CAM)
    except Exception:
        body = None

    index_rows = []
    cells = []       # one dict per jump: crops + overlay data + label

    for r in jumps.itertuples():
        onset = int(r.onset_frame)
        frames = [onset + int(round(o * L.FPS)) for o in OFFSETS_S]

        # centre the crop on the mean head position over the triplet
        # (fall backs: body centroid -> frame centre)
        hxs, hys = [], []
        for f in frames + [onset]:
            if (r.jump, f) in kc.index:
                row = kc.loc[(r.jump, f)]
                hx, hy = row.get("head_x"), row.get("head_y")
                if np.isfinite(hx) and np.isfinite(hy):
                    hxs.append(float(hx)); hys.append(float(hy))
        cen = (float(np.mean(hxs)), float(np.mean(hys))) if hxs else None
        if cen is None and body is not None:
            f = np.clip(onset, 0, len(body["cx"]) - 1)
            cen = (float(body["cx"][f]), float(L.IMG - 1 - body["y_up"][f]))
        if cen is None:
            cen = (L.IMG / 2, L.IMG / 2)
        x0, y0 = crop_box(*cen)

        trip = []
        for f, tag, off in zip(frames, TAGS, OFFSETS_S):
            img = grab_frame(CAM, f)
            crop = img[y0:y0 + CROP, x0:x0 + CROP]
            ov = kc.loc[(r.jump, f)].to_dict() if (r.jump, f) in kc.index else {}
            trip.append(dict(crop=crop, ov=ov, x0=x0, y0=y0, frame=f, tag=tag))

            # individual PNGs (overlay + clean)
            for clean in (False, True):
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(crop, cmap="gray", vmin=0, vmax=255,
                          extent=[x0, x0 + CROP, y0 + CROP, y0])
                if not clean:
                    draw_overlay(ax, ov, x0, y0)
                ax.set_xlim(x0, x0 + CROP); ax.set_ylim(y0 + CROP, y0)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{r.jump}  {r.cls}   t={off:+.2f}s   f{f}", fontsize=9)
                fig.tight_layout()
                suff = "_clean" if clean else ""
                fig.savefig(os.path.join(
                    OUTDIR, f"{r.jump}_cam{CAM}_{tag}{suff}.png"), dpi=110)
                plt.close(fig)

            index_rows.append(dict(
                jump=r.jump, cls=r.cls, ballistic=r.ballistic,
                onset_frame=onset, onset_t_s=r.onset_t_s,
                offset_s=off, abs_frame=f, abs_t_s=round(f / L.FPS, 3),
                crop_x0=x0, crop_y0=y0, crop_side=CROP))

        cells.append(dict(jump=r.jump, cls=r.cls, ballistic=r.ballistic,
                          onset=onset, onset_t=r.onset_t_s,
                          pk=r.peak_head_speed_pxs, note=str(r.note),
                          cov=getattr(r, f"cam{CAM}_cov_pct"), trip=trip))

    pd.DataFrame(index_rows).to_csv(
        os.path.join(OUTDIR, f"index_cam{CAM}.csv"), index=False)

    # ---- montages: 8 jumps / figure, 2 cols x 4 rows, cell = 3-frame strip ----
    ngroups = int(np.ceil(len(cells) / PER_FIG))
    for g in range(ngroups):
        grp = cells[g * PER_FIG:(g + 1) * PER_FIG]
        nrow = min(4, int(np.ceil(len(grp) / 2)))
        height = 2.9 * nrow + 1.0
        fig, axes = plt.subplots(nrow, 6, figsize=(15.4, height), squeeze=False)
        for ax in axes.ravel():
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        used = set()
        for idx, c in enumerate(grp):
            rr, cc = idx % nrow, idx // nrow        # fill column-major: col0 then col1
            base = cc * 3
            used.update({(rr, base + k) for k in range(3)})
            for k, fr in enumerate(c["trip"]):
                ax = axes[rr][base + k]
                x0, y0 = fr["x0"], fr["y0"]
                ax.imshow(fr["crop"], cmap="gray", vmin=0, vmax=255,
                          extent=[x0, x0 + CROP, y0 + CROP, y0])
                draw_overlay(ax, fr["ov"], x0, y0)
                ax.set_xlim(x0, x0 + CROP); ax.set_ylim(y0 + CROP, y0)
                ax.set_title(LBL[k], fontsize=8.5,
                             color="k" if k < 2 else "#b00020")
                for s in ax.spines.values():
                    s.set_visible(True)
                    s.set_color("#b00020" if k == 2 else "#cccccc")
                    s.set_linewidth(1.6 if k == 2 else 0.8)
            head = (f"{c['jump']}   {c['cls']}   ballistic={c['ballistic'] or '?'}"
                    f"    onset f{c['onset']} ({c['onset_t']:.2f} s)   "
                    f"peak {c['pk']:.0f} px/s")
            axes[rr][base + 1].annotate(
                head, xy=(0.5, 1.0), xycoords="axes fraction",
                xytext=(0, 26), textcoords="offset points",
                ha="center", va="bottom", fontsize=9.2, fontweight="bold")
        # blank any unused cells
        for rr in range(nrow):
            for col in range(6):
                if (rr, col) not in used:
                    axes[rr][col].axis("off")
        fig.suptitle(
            f"Cam{CAM} frames 0.10 s / 0.05 s / 0 s before jump onset   "
            f"(onset t=0 = frame of peak Cam2 head-centroid speed in the event)   "
            f"[figure {g + 1} of {ngroups}]\n"
            "dots = snout/eye/ear/head_top/nape (filled = above 0.6 DLC confidence)   "
            "white ring = head centroid   white arrow = nape\u2192snout head axis",
            fontsize=10.5)
        fig.tight_layout(rect=[0, 0, 1, 1 - 0.62 / height])
        fig.subplots_adjust(wspace=0.06, hspace=0.62)
        out = os.path.join(FIGDIR, f"fig_onset_triplets_cam{CAM}_g{g + 1}.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print("wrote", out, flush=True)


def draw_overlay(ax, ov, x0=0, y0=0):
    if not ov:
        return

    def inside(x, y):
        return (x is not None and np.isfinite(x) and np.isfinite(y)
                and x0 <= x <= x0 + CROP and y0 <= y <= y0 + CROP)

    for bp, col in KP_COLORS:
        x, y = ov.get(f"{bp}_x"), ov.get(f"{bp}_y")
        xr, yr = ov.get(f"{bp}_x_raw"), ov.get(f"{bp}_y_raw")
        if inside(x, y):
            ax.plot(x, y, "o", ms=5, color=col, mec="k", mew=0.5, zorder=5)
        elif inside(xr, yr):
            ax.plot(xr, yr, "o", ms=5, mfc="none", mec=col, mew=1.0, zorder=5)
    hx, hy = ov.get("head_x"), ov.get("head_y")
    if inside(hx, hy):
        ax.plot(hx, hy, "o", ms=11, mfc="none", mec="white", mew=1.4, zorder=6)
    nx, ny = ov.get("nape_x"), ov.get("nape_y")
    sx, sy = ov.get("snout_x"), ov.get("snout_y")
    if inside(nx, ny) and inside(sx, sy):
        v = np.array([sx - nx, sy - ny], float)
        n = np.linalg.norm(v)
        if n > 1:
            tip = np.array([sx, sy]) + v / n * 55
            ax.annotate("", xy=tuple(tip), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="-|>", color="white", lw=1.6),
                        zorder=6)


if __name__ == "__main__":
    main()
