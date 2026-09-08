"""
v4 rerun of fig09 (head-axis aiming at the landing targets) + a worked-example
"snip" for the up->up / forepaw-touchdown case, drawn on real Cam2 frames in the
style of fig16b.

Outputs (all under v4/figures/):
    fig09_target_aiming.png          target = MEASURED head landing position
    fig09b_edge_aiming.png           target = landing-platform front edge (up->up)
    fig09c_paw_aiming.png            target = MANUAL forepaw touchdown point
    fig09d_paw_aiming_examples.png   NEW: 3 up->up jumps, Cam2 frames at
        t = -0.6..+0.15 s, the nape->snout head axis vs the direction to the
        forepaw touchdown point, plus the per-jump aiming-error trace.
        2 jumps close to the group's approach aiming error + 1 outlier.

Runs on the v4 tables (v3_05/07 dropped, v3_06/12 re-timed); reuses
v4_figures.build_W and the v3_analysis plotting functions unchanged.

    E:\\TreeShrew_Jumping\\analysis\\dlc\\py310\\python.exe v4_fig09.py [j_rep1 j_rep2 j_outlier]
"""
import os
import sys
import numpy as np
import pandas as pd
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# E: figures occasionally hold a transient lock (viewer / indexer) -> EINVAL on
# save. Retry a few times before giving up.
_orig_savefig = plt.Figure.savefig


def _savefig_retry(self, fname, *a, **k):
    for attempt in range(8):
        try:
            return _orig_savefig(self, fname, *a, **k)
        except OSError:
            if attempt == 7:
                raise
            time.sleep(0.7)


plt.Figure.savefig = _savefig_retry

import v3_lib as L
import v3_analysis as A
import v3_extra as X
from v3_build import GEOM
import v4_figures as V4F

V4 = os.path.join(L.AN, "v4")
FIG4 = os.path.join(V4, "figures")
os.makedirs(FIG4, exist_ok=True)
A.FIG = FIG4                       # fig_target saves to A.FIG

# use the re-labelled touchdown points if the GUI (v4_paw_gui.py) has written them
_pc_v4 = os.path.join(V4, "paw_contacts_v4.csv")
if os.path.exists(_pc_v4):
    A.PAW_CONTACTS_PATH = _pc_v4
    print(f"paw contacts: {os.path.relpath(_pc_v4, L.AN)}")
else:
    print("paw contacts: v3/paw_contacts.csv (run v4_paw_gui.py to re-label)")

SNAP_TIMES = [-0.6, -0.4, -0.2, 0.0, 0.15]
KP_STYLE = [("snout", "red", 5), ("eye", "yellow", 3.5), ("ear", "lime", 3.5),
            ("head_top", "cyan", 3.5), ("nape", "magenta", 5)]


# --------------------------------------------------------------------------- #
def paw_error_raw(d, paw_raw):
    """Signed head-axis aiming error vs the forepaw touchdown point, in the raw
    (un-mirrored) Cam2 frame so it matches the drawn frames. |value| equals
    A.target_error(w, 'paw'); only the sign convention differs (frame-native:
    +ve = head axis points ABOVE the straight-to-target line)."""
    px, pz = paw_raw
    dx = px - d.head_x.to_numpy()
    dz = pz - d.head_z.to_numpy()
    return L.wrap180(d.ang_nape_snout.to_numpy() - L.ang_deg(dx, dz))


def pick_examples(W, grid):
    """2 up->up jumps nearest the group's median approach aiming error + the 1
    with the largest approach aiming error (the outlier)."""
    rows = []
    for j, w in W.items():
        if w["row"].cls != "up->up" or not np.isfinite(w["paw_raw"][0]):
            continue
        err = A.target_error(w, grid, "paw")
        cov = np.isfinite(err[(grid >= -1.0) & (grid < 0.0)]).mean()
        if cov < 0.5:
            continue
        appr = float(np.nanmean(np.abs(err[(grid >= -0.6) & (grid < -0.1)])))
        rows.append(dict(jump=j, appr_abs_err=appr, approach_cov=round(cov, 2),
                         paw_conf=w["paw_conf"]))
    tab = pd.DataFrame(rows).sort_values("appr_abs_err").reset_index(drop=True)
    med = tab.appr_abs_err.median()
    near = tab.reindex(tab.appr_abs_err.sub(med).abs().sort_values().index)
    rep = list(near.jump.iloc[:2])
    outlier = tab.jump.iloc[-1]
    return rep, outlier, tab


# --------------------------------------------------------------------------- #
def _crop_box(d, idx, paw_raw, aspect):
    """Bounding box round the head keypoints (all shown frames) + the paw point,
    padded, then grown to the panel's width:height `aspect` so the frame fills
    the axes instead of floating in white space."""
    xs, ys = [paw_raw[0]], [L.IMG - 1 - paw_raw[1]]
    for i in idx:
        for bp in ("nape", "snout", "eye"):
            bx, bz = d[f"{bp}_x"].to_numpy()[i], d[f"{bp}_z"].to_numpy()[i]
            if np.isfinite(bx):
                xs.append(bx)
                ys.append(L.IMG - 1 - bz)
    x0, x1 = min(xs) - 95, max(xs) + 95
    y0, y1 = min(ys) - 65, max(ys) + 65
    w, h = x1 - x0, y1 - y0
    if w / h < aspect:                       # too tall -> widen
        pad = (aspect * h - w) / 2
        x0, x1 = x0 - pad, x1 + pad
    else:                                    # too wide -> heighten
        pad = (w / aspect - h) / 2
        y0, y1 = y0 - pad, y1 + pad
    return x0, x1, y0, y1


def _draw_frame(ax, d, i, paw_raw, box):
    x0, x1, y0, y1 = box
    fr = int(d.abs_frame.to_numpy()[i])
    ax.imshow(X.grab_frame(2, fr / L.FPS), cmap="gray", vmin=0, vmax=255)

    nx = d.nape_x.to_numpy()[i]
    nz = d.nape_z.to_numpy()[i]
    ang = d.ang_nape_snout.to_numpy()[i]
    px, pz = paw_raw
    py_img = L.IMG - 1 - pz

    # forepaw touchdown target
    ax.plot(px, py_img, "+", ms=15, mew=2.2, color="cyan")
    ax.plot(px, py_img, "o", ms=13, mfc="none", mec="cyan", mew=1.3)

    err = np.nan
    if np.isfinite(nx) and np.isfinite(ang):
        ny_img = L.IMG - 1 - nz
        reach = float(np.hypot(px - nx, pz - nz))
        # direction straight to the target (green) and the actual head axis (blue)
        ax.plot([nx, px], [ny_img, py_img], "-", color="lime", lw=1.3, alpha=0.9)
        ex = nx + reach * np.cos(np.deg2rad(ang))
        ey = ny_img - reach * np.sin(np.deg2rad(ang))
        ax.plot([nx, ex], [ny_img, ey], "-", color="white", lw=1.6)
        ax.plot([nx, ex], [ny_img, ey], "--", color="tab:blue", lw=1.3)
        err = float(L.wrap180(ang - L.ang_deg(px - nx, pz - nz)))

    for bp, col, ms in KP_STYLE:
        bx, bz = d[f"{bp}_x"].to_numpy()[i], d[f"{bp}_z"].to_numpy()[i]
        if np.isfinite(bx):
            ax.plot(bx, L.IMG - 1 - bz, "o", ms=ms, color=col, mec="k", mew=.4)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    return err


def fig_paw_examples(W, grid, rep, outlier, tab):
    picks = [(rep[0], "representative"), (rep[1], "representative"),
             (outlier, "outlier")]
    nrow, ncol = len(picks), len(SNAP_TIMES) + 1
    pw, ph = 3.35, 2.5
    aspect = pw / ph
    fig, axes = plt.subplots(nrow, ncol, figsize=(pw * ncol, ph * nrow),
                             squeeze=False)

    for ri, (j, tag) in enumerate(picks):
        w = W[j]
        d = w["raw"]
        sgn = w["sgn"]
        paw_raw = w["paw_raw"]
        t = d.t.to_numpy()
        idx = [int(np.nanargmin(np.abs(t - tt))) for tt in SNAP_TIMES]
        box = _crop_box(d, idx, paw_raw, aspect)

        for ci, (tt, i) in enumerate(zip(SNAP_TIMES, idx)):
            ax = axes[ri][ci]
            err = _draw_frame(ax, d, i, paw_raw, box)
            ttl = f"t = {tt:+.2f} s"
            if np.isfinite(err):
                ttl += f"\naim err = {err:+.0f} deg"
            ax.set_title(ttl, fontsize=8)

        # ---- aiming-error trace (last column) ---- #
        ax = axes[ri][-1]
        err_t = paw_error_raw(d, paw_raw)
        dist = np.hypot(paw_raw[0] - d.head_x.to_numpy(),
                        paw_raw[1] - d.head_z.to_numpy())
        keep = (dist >= A.TARGET_MIN_DIST) & (t <= w["t_land"] + 0.15)
        et = np.where(keep, err_t, np.nan)
        ax.plot(t, et, "-", color=A.CLASS_COLORS["up->up"], lw=1.5)
        mA = (t >= -0.6) & (t < -0.1) & np.isfinite(et)
        if mA.any():
            lo, hi = np.nanpercentile(et[mA], [5, 95])
            ax.axhspan(lo, hi, color=A.CLASS_COLORS["up->up"], alpha=0.15,
                       label=f"approach 5-95 ({hi - lo:.0f} deg)")
        ax.axhline(0, color="tab:green", lw=1.4)
        ax.axvline(0, color="k", ls="--", lw=1)
        for tt, i in zip(SNAP_TIMES, idx):
            if np.isfinite(et[i]):
                ax.plot(tt, et[i], "o", ms=6, color="k", zorder=5)
        ax.set_xlim(-1.0, 0.5)
        ax.set_ylim(-100, 100)
        ax.set_xlabel("time from onset (s)")
        ax.set_ylabel("aiming error (deg)")
        ax.set_title("head-axis aiming error vs the forepaw\n"
                     "target (the number fig09c tracks)", fontsize=8)
        ax.legend(fontsize=6.5, loc="upper left")
        ax.grid(True, alpha=0.3)

        r = tab.set_index("jump").loc[j]
        axes[ri][0].set_ylabel(
            f"{j}  [{tag}]\n"
            f"{'right' if sgn > 0 else 'left'}ward up->up\n"
            f"approach |err| {r.appr_abs_err:.0f} deg\n"
            f"paw conf: {r.paw_conf}", fontsize=8.5)

    fig.text(
        0.5, 0.055,
        "magenta = nape    red = snout    "
        "white/blue dashed = nape->snout head axis (extended to target range)    "
        "green = straight line from the head to the forepaw touchdown point\n"
        "cyan + = manually annotated forepaw touchdown point (Cam2 px, ~+-20 px)    "
        "aim err = signed angle between the blue and green lines "
        "(0 = head pointed straight at the target)",
        ha="center", fontsize=7.5)

    med = tab.appr_abs_err.median()
    fig.suptitle(
        "fig09d  Worked examples of the fig09c head aiming, on real Cam2 frames "
        "(up->up gap crossings, v4 jump set).\n"
        "Blue/white line = nape->snout head axis;  green line = straight to the "
        f"forepaw touchdown point.  Group median approach |aiming error| "
        f"= {med:.0f} deg:  rows 1-2 sit near it, row 3 is the outlier.",
        fontsize=10)
    fig.tight_layout(rect=(0, 0.085, 1, 0.95))
    out = os.path.join(FIG4, "fig09d_paw_aiming_examples.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(out, L.AN)}")


# --------------------------------------------------------------------------- #
def main():
    jumps, kp = V4F.load_v4()
    grid = np.arange(-A.PRE_S, A.POST_S + 1e-9, 0.01)
    W = V4F.build_W(jumps, kp)
    print(f"{len(W)} jumps: " + ", ".join(f"{j}[{W[j]['row'].cls}]" for j in W))

    by_cls = {c: [j for j in W if W[j]["row"].cls == c] for c in A.CLASS_COLORS}
    by_cls = {c: v for c, v in by_cls.items() if v}
    print("by class: " + ", ".join(f"{c} n={len(v)}" for c, v in by_cls.items()))

    # ---- fig09 / 09b / 09c : the three aiming figures on the v4 set ---------- #
    A.fig_target(grid, W, by_cls, which="target", fname="fig09_target_aiming.png",
                 subtitle="Target = the MEASURED position where the head settles "
                          "after landing.")
    A.fig_target(grid, W, by_cls, which="edge", fname="fig09b_edge_aiming.png",
                 classes=("up->up",),
                 subtitle="Target = the LANDING PLATFORM'S FRONT EDGE. up->up only "
                          "(for up->down the edge is behind the animal).")
    A.fig_target(grid, W, by_cls, which="paw", fname="fig09c_paw_aiming.png",
                 subtitle="Target = the MANUALLY ANNOTATED forepaw touchdown point "
                          "(~+-20 px; DLC tracks head only; dotted = low-confidence jump).")
    for f in ("fig09_target_aiming.png", "fig09b_edge_aiming.png",
              "fig09c_paw_aiming.png"):
        print(f"  wrote figures/{f}")

    # ---- fig09d : worked examples ------------------------------------------- #
    if len(sys.argv) >= 4:
        rep, outlier = sys.argv[1:3], sys.argv[3]
        _, _, tab = pick_examples(W, grid)
    else:
        rep, outlier, tab = pick_examples(W, grid)
    print("\nup->up paw-aiming candidates (approach window -0.6..-0.1 s):")
    print(tab.to_string(index=False))
    print(f"\npicked representative = {rep}, outlier = {outlier}")
    fig_paw_examples(W, grid, rep, outlier, tab)

    print(f"\nfigures -> {FIG4}")


if __name__ == "__main__":
    main()
