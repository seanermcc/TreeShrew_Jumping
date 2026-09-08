"""
Forepaw touchdown check sheet.

For every v4 jump, a strip of Cam2 frames from jump onset to +0.40 s (the whole
landing), cropped on where the head actually ends up, with the CURRENT paw
annotation (cyan +) drawn on for reference. The point is to see, per jump, the
frame where the forepaw actually plants and how far the current mark is from it
-- without trusting the (possibly wrong) annotated contact frame.

    E:\\TreeShrew_Jumping\\analysis\\dlc\\py310\\python.exe v4_paw_snaps.py

Output: v4/figures/fig09e_paw_contact_frames_{1,2}.png
"""
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_orig_savefig = plt.Figure.savefig


def _savefig_retry(self, fname, *a, **k):          # E: transient file locks
    for attempt in range(8):
        try:
            return _orig_savefig(self, fname, *a, **k)
        except OSError:
            if attempt == 7:
                raise
            time.sleep(0.7)


plt.Figure.savefig = _savefig_retry

import v3_lib as L
import v3_extra as X

V4 = os.path.join(L.AN, "v4")
FIG4 = os.path.join(V4, "figures")
os.makedirs(FIG4, exist_ok=True)

DFRAMES = list(range(0, 41, 5))      # onset .. +0.40 s, every 50 ms  (9 frames)
ROWS_PER_FIG = 7


PC_PATH = os.path.join(V4, "paw_contacts_v4.csv")
if not os.path.exists(PC_PATH):
    PC_PATH = os.path.join(L.V3, "paw_contacts.csv")


def load():
    jumps = pd.read_csv(os.path.join(V4, "v4_jumps.csv"))
    print(f"annotations: {os.path.relpath(PC_PATH, L.AN)}")
    pc = pd.read_csv(PC_PATH)
    if "status" in pc.columns:
        pc = pc[~pc["status"].fillna("ok").isin(["dropped", "none", "no_contact"])]
    kp = pd.read_csv(os.path.join(V4, "_keypoints_v4.csv"))
    order = {j: i for i, j in enumerate(jumps.jump)}
    pc = pc[pc.jump.isin(order)].copy()
    pc["_o"] = pc.jump.map(order)
    pc = pc.sort_values("_o").reset_index(drop=True)
    pc = pc.drop(columns=[c for c in ("onset_frame", "cls", "dframe") if c in pc.columns])
    pc = pc.merge(jumps[["jump", "cls", "onset_frame"]], on="jump", how="left")
    missing = [j for j in jumps.jump if j not in set(pc.jump)]
    return pc, kp, missing


def landing_center(kp, r):
    """Median Cam2 head position over t = +0.20 .. +0.45 s -- where the animal
    settles after the jump; used to frame the crop."""
    o = int(r.onset_frame)
    k = kp[(kp.cam == 2) & (kp.abs_frame >= o + 20) & (kp.abs_frame <= o + 45)]
    if len(k) and np.isfinite(k.head_x).any():
        return float(np.nanmedian(k.head_x)), float(np.nanmedian(k.head_y))
    return float(r.paw_x_img), float(r.paw_y_img)


def head_kps(kp, frame):
    k = kp[(kp.cam == 2) & (kp.abs_frame == frame)]
    return k.iloc[0] if len(k) else None


def draw_row(axes, r, kp):
    o = int(r.onset_frame)
    cf = int(r.contact_frame)
    px, py = float(r.paw_x_img), float(r.paw_y_img)
    hx, hy = landing_center(kp, r)
    if str(r.surface) == "floor":
        # DLC loses the head during the drop, so frame on the annotation
        cx, cy, cw, ch = px, py - 90, 580, 460
    else:
        cx, cy, cw, ch = 0.5 * (hx + px), 0.5 * (hy + py), 480, 340
    x0 = int(np.clip(cx - cw / 2, 0, L.IMG - cw))
    y0 = int(np.clip(cy - ch / 2, 0, L.IMG - ch))
    CROP_W, CROP_H = cw, ch

    for ax, df in zip(axes, DFRAMES):
        fr = o + df
        img = X.grab_frame(2, fr / L.FPS)
        ax.imshow(img[y0:y0 + CROP_H, x0:x0 + CROP_W], cmap="gray", vmin=0, vmax=255,
                  extent=[x0, x0 + CROP_W, y0 + CROP_H, y0])
        ax.plot(px, py, "+", ms=15, mew=1.8, color="cyan")
        ax.plot(px, py, "o", ms=13, mfc="none", mec="cyan", mew=1.0)
        row = head_kps(kp, fr)
        if row is not None:
            for bp, col in [("snout", "red"), ("eye", "yellow"), ("nape", "magenta")]:
                if np.isfinite(row[f"{bp}_x"]):
                    ax.plot(row[f"{bp}_x"], row[f"{bp}_y"], "o", ms=4, color=col,
                            mec="k", mew=.3)
            if np.isfinite(row["nape_x"]) and np.isfinite(row["snout_x"]):
                ax.plot([row["nape_x"], row["snout_x"]], [row["nape_y"], row["snout_y"]],
                        "-", color="white", lw=1.0)
        ax.set_xlim(x0, x0 + CROP_W)
        ax.set_ylim(y0 + CROP_H, y0)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        tag = f"+{df / L.FPS * 1000:.0f} ms"
        if abs(fr - cf) <= 1:
            tag += "  <- annotated"
        ax.set_title(f"f{fr}   {tag}", fontsize=7.5)

    axes[0].set_ylabel(
        f"{r.jump}   [{r.cls}]\n{r.surface}  conf:{r.confidence}\n"
        f"annot = onset{cf - o:+d} fr\n({r.paw_x_img:.0f}, {r.paw_y_img:.0f})",
        fontsize=8, rotation=90, labelpad=6)


def main():
    pc, kp, missing = load()
    print(f"{len(pc)} annotated jumps; no annotation (no gap crossing): {missing}\n")
    print("jump    cls        annot(x,y)      annot_dframe   head-landing(x,y)")
    for _, r in pc.iterrows():
        cx, cy = landing_center(kp, r)
        print(f"{r.jump:7s} {r.cls:10s} ({r.paw_x_img:6.0f},{r.paw_y_img:6.0f})  "
              f"onset{int(r.contact_frame) - int(r.onset_frame):+4d}      "
              f"({cx:6.0f},{cy:6.0f})")

    chunks = [pc.iloc[i:i + ROWS_PER_FIG] for i in range(0, len(pc), ROWS_PER_FIG)]
    for ci, chunk in enumerate(chunks, 1):
        nrow = len(chunk)
        fig, axes = plt.subplots(nrow, len(DFRAMES),
                                 figsize=(1.95 * len(DFRAMES), 2.3 * nrow),
                                 squeeze=False)
        for ax_row, (_, r) in zip(axes, chunk.iterrows()):
            draw_row(ax_row, r, kp)
        fig.suptitle(
            "fig09e  Forepaw touchdown check -- Cam2, jump onset to +0.40 s "
            f"(sheet {ci}/{len(chunks)}).  Crop follows where the head lands.\n"
            f"cyan + = current annotation ({os.path.basename(PC_PATH)}), identical on "
            "every frame.  red=snout  yellow=eye  magenta=nape  white=head axis.\n"
            "Find the frame where the forepaw plants; note how far the cyan mark "
            "is from it (and whether the annotated frame, marked, is even close).",
            fontsize=9.5)
        fig.tight_layout(rect=(0, 0, 1, 0.95 if nrow > 3 else 0.88))
        out = os.path.join(FIG4, f"fig09e_paw_contact_frames_{ci}.png")
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  wrote {os.path.relpath(out, L.AN)}")


if __name__ == "__main__":
    main()
