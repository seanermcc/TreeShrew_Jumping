"""
v3 jump analysis: speed, head orientation, head-vs-body, and target-aiming.

Everything is aligned to JUMP ONSET (t = 0), defined as the frame of peak Cam2
head-centroid speed inside the confirmed event window (see v3_build.py), and
runs over -1 s .. +3 s.

Units are full-resolution image pixels and px/s. There is NO camera calibration
for this rig, so nothing is metric: relative magnitudes, shapes and angles in
the Cam2 sagittal plane are meaningful, absolute distances are not.

Direction normalisation: the animal travels frame-left on some jumps and
frame-right on others. For every pooled ORIENTATION result the x axis is
mirrored so that all jumps travel to the RIGHT (+x); mirrored jumps are marked
in v3_summary.csv. Speeds are direction-invariant and are never mirrored.

Outputs -> v3/figures/*.png, v3/v3_summary.csv, v3/v3_epoch_stats.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import v3_lib as L
from v3_build import GEOM

FIG = os.path.join(L.V3, "figures")
PRE_S, POST_S = 1.0, 3.0
EPOCHS = [("pre", -1.0, 0.0), ("jump", 0.0, 0.2), ("post", 0.2, 3.0)]
CLASS_COLORS = {"up->up": "tab:blue", "up->down": "tab:red", "down->up": "tab:green"}

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})


# --------------------------------------------------------------------------- #
# per-jump window assembly
# --------------------------------------------------------------------------- #
def window(kp, body, jrow, cam=2):
    """Tidy -1..+3 s window for one jump: cleaned tracks, speeds, angles."""
    k = kp[(kp["jump"] == jrow.jump) & (kp.cam == cam)].sort_values("abs_frame")
    if not len(k):
        return None
    f = k.abs_frame.to_numpy()
    t = (f - jrow.onset_frame) / L.FPS
    m = (t >= -PRE_S - 1e-9) & (t <= POST_S + 1e-9)
    if m.sum() < 10:
        return None
    k, t, f = k[m], t[m], f[m]

    d = pd.DataFrame({"t": np.round(t, 2), "abs_frame": f})
    for bp in L.BODYPARTS:
        d[f"{bp}_x"] = L.clean_series(k[f"{bp}_x"].to_numpy())
        d[f"{bp}_z"] = L.clean_series(k[f"{bp}_y_up"].to_numpy())
    d["head_x"] = L.clean_series(k.head_x.to_numpy())
    d["head_z"] = L.clean_series(k.head_y_up.to_numpy())
    d["head_raw_ok"] = np.isfinite(k.head_x.to_numpy())
    d["head_speed"] = L.speed(d.head_x, d.head_z)
    d["snout_speed"] = L.speed(d.snout_x, d.snout_z)

    if cam == 2:
        bx, bz = L.body_at(body, f)
        d["body_x"], d["body_z"] = bx, bz
        d["body_speed"] = L.speed(bx, bz)
        d["rel_speed"] = d["head_speed"] - d["body_speed"]

    d["ang_nape_snout"] = L.ang_deg(d.snout_x - d.nape_x, d.snout_z - d.nape_z)
    d["ang_eye_snout"] = L.ang_deg(d.snout_x - d.eye_x, d.snout_z - d.eye_z)
    for a, b in [("nape", "snout"), ("nape", "eye"), ("nape", "head_top"),
                 ("nape", "ear"), ("eye", "snout")]:
        d[f"d_{a}_{b}"] = np.hypot(d[f"{b}_x"] - d[f"{a}_x"], d[f"{b}_z"] - d[f"{a}_z"])
    return d


def travel_sign(d):
    """+1 if the animal moves frame-right across the jump, -1 if frame-left."""
    pre = d.head_x[(d.t >= -0.30) & (d.t <= 0.0)].median()
    post = d.head_x[(d.t >= 0.30) & (d.t <= 1.2)].median()
    if not np.isfinite(pre) or not np.isfinite(post):
        pre, post = d.head_x[d.t < 0].median(), d.head_x[d.t > 0].median()
    s = np.sign(post - pre)
    return 1.0 if s == 0 or not np.isfinite(s) else float(s)


def mirror(d, sgn):
    """Mirror x so travel is rightward; reflect angles about the vertical."""
    if sgn > 0:
        return d
    d = d.copy()
    for c in d.columns:
        if c.endswith("_x"):
            d[c] = -d[c]
    for c in ("ang_nape_snout", "ang_eye_snout"):
        d[c] = L.wrap180(180.0 - d[c])
    return d


def landing_target(d, jrow):
    """Where the animal actually ended up: measured head (x, z) after it settles.

    Deliberately MEASURED rather than derived from the arena geometry. The
    animal lies prone on a platform (head ~20-100 px above the surface) but
    stands upright on the floor (~185 px), so "surface height + a constant"
    would put the target in the wrong place by ~100 px on the down-jumps.
    """
    post = d[(d.t > 0.05) & np.isfinite(d.head_speed)]
    quiet = post[post.head_speed < 250]
    if len(quiet):
        tq = quiet.t.to_numpy()
        runs = np.split(tq, np.where(np.diff(tq) > 0.03)[0] + 1)
        run = next((r for r in runs if len(r) >= 15), runs[0])
        t_land = float(run[0])
    else:
        t_land = float(d.t.iloc[-1])
    seg = d[(d.t >= t_land) & (d.t <= t_land + 0.4)]
    if not len(seg):
        return np.nan, np.nan, t_land
    return (float(np.nanmedian(seg.head_x)), float(np.nanmedian(seg.head_z)), t_land)


def crosses_gap(d_raw):
    """True if the head actually passes over the gap between the platforms."""
    x = d_raw.head_x.to_numpy()
    inside = np.isfinite(x) & (x > GEOM["left_edge_x"]) & (x < GEOM["right_edge_x"])
    return bool(inside.sum() >= 3)


# override to point at a re-labelled file (e.g. v4/paw_contacts_v4.csv)
PAW_CONTACTS_PATH = os.path.join(L.V3, "paw_contacts.csv")


def load_paw_contacts():
    """Manually annotated forepaw first-contact point per jump (Cam2, image px).

    The DLC model tracks only the head, so the touchdown point was read off
    frame-by-frame from the video (see paw_contacts.csv, ~+-20 px, +-3 frames;
    a `confidence` column flags the shakier ones -- slow crawls have no discrete
    contact). Rows flagged `status == "no_contact"` are skipped. Returns
    {jump: (x_img, y_up, confidence)}.
    """
    p = PAW_CONTACTS_PATH
    if not os.path.exists(p):
        return {}
    df = pd.read_csv(p)
    if "status" in df.columns:
        df = df[~df["status"].fillna("ok").isin(["no_contact", "dropped", "none"])]
    return {r.jump: (float(r.paw_x_img), L.IMG - 1 - float(r.paw_y_img), r.confidence)
            for r in df.itertuples()}


def edge_target(sgn):
    """The landing platform's front edge, expressed in the mirrored frame.

    In the mirrored frame travel is always +x, so the edge the animal lands on
    is simply the further-forward of the two mirrored edges.
    Only meaningful for jumps that actually cross the gap.
    """
    lx, rx = GEOM["left_edge_x"], GEOM["right_edge_x"]
    return float(max(sgn * lx, sgn * rx)), GEOM["plat_surface_y_up"]


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def resample(d, col, grid):
    t = d.t.to_numpy()
    v = pd.to_numeric(d[col], errors="coerce").to_numpy()
    ok = np.isfinite(t) & np.isfinite(v)
    if ok.sum() < 3:
        return np.full(len(grid), np.nan)
    tt, vv = t[ok], v[ok]
    out = np.interp(grid, tt, vv, left=np.nan, right=np.nan)
    for i, g in enumerate(np.diff(tt)):        # never bridge >80 ms dropouts
        if g > 0.08:
            out[(grid > tt[i]) & (grid < tt[i + 1])] = np.nan
    return out


def epoch_means(d, col):
    v = pd.to_numeric(d[col], errors="coerce")
    return {n: (float(np.nanmean(v[(d.t >= a) & (d.t < b)]))
                if ((d.t >= a) & (d.t < b)).any() else np.nan)
            for n, a, b in EPOCHS}


def sem(a, axis=0):
    a = np.asarray(a, float)
    n = np.sum(np.isfinite(a), axis=axis)
    with np.errstate(invalid="ignore"):
        sd = np.nanstd(a, axis=axis, ddof=1)
    return np.where(n > 1, sd / np.sqrt(np.maximum(n, 1)), np.nan)


def bars(ax, groups, labels, colors, title, ylab):
    ne = len(EPOCHS)
    w = 0.8 / len(groups)
    rng = np.random.default_rng(0)
    for gi, (g, lab, col) in enumerate(zip(groups, labels, colors)):
        m, e = np.nanmean(g, axis=0), sem(g, axis=0)
        n = np.sum(np.isfinite(g), axis=0)
        xs = np.arange(ne) + gi * w - 0.4 + w / 2
        ax.bar(xs, m, w * 0.9, yerr=e, capsize=4, color=col, alpha=0.85,
               label=f"{lab} (n={int(np.nanmax(n))})", edgecolor="black", linewidth=0.5)
        for x, gv in zip(xs, np.asarray(g).T):
            gv = gv[np.isfinite(gv)]
            ax.plot(np.full(len(gv), x) + rng.uniform(-w * .12, w * .12, len(gv)),
                    gv, "o", ms=2.5, color="black", alpha=0.45, zorder=3)
    ax.set_xticks(range(ne))
    ax.set_xticklabels([f"{n}\n[{a:+.1f},{b:+.1f}) s" for n, a, b in EPOCHS])
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.legend(fontsize=7)


def unwrap_deg(mat):
    """Kept as the identity: these head angles live around -140..0 deg and never
    cross +-180, so unwrapping buys nothing -- and unwrapping each row then
    re-centring it makes every row's absolute level arbitrary, which showed up
    as a spurious monotonic drift in the pooled mean. Use circular statistics
    on the wrapped values instead."""
    return np.asarray(mat, float)


# --------------------------------------------------------------------------- #
def main():
    os.makedirs(FIG, exist_ok=True)
    jumps = pd.read_csv(os.path.join(L.V3, "v3_jumps.csv"))
    # one jump has no manual ballistic label; keep it as "" not NaN
    jumps["ballistic"] = jumps["ballistic"].fillna("").astype(str)
    kp = L.build_keypoints_v3()
    body = L.load_body(2)
    grid = np.arange(-PRE_S, POST_S + 1e-9, 0.01)

    paws = load_paw_contacts()

    W, meta = {}, []
    for r in jumps.itertuples():
        d = window(kp, body, r, cam=2)
        if d is None:
            print(f"  !! no window for {r.jump}")
            continue
        sgn = travel_sign(d)
        dm = mirror(d, sgn)
        xt, zt, t_land = landing_target(dm, r)
        gap = crosses_gap(d)
        # paw touchdown: keep both the raw image point and the mirrored one
        if r.jump in paws:
            px, pz, pconf = paws[r.jump]
            paw_raw = (px, pz)
            paw = (-px if sgn < 0 else px, pz)
        else:
            paw_raw = paw = (np.nan, np.nan)
            pconf = ""
        W[r.jump] = dict(raw=d, m=dm, sgn=sgn, target=(xt, zt), t_land=t_land,
                         row=r, gap=gap, edge=edge_target(sgn) if gap else (np.nan, np.nan),
                         paw=paw, paw_raw=paw_raw, paw_conf=pconf)
        meta.append(dict(
            jump=r.jump, src_clips=r.src_clips, cls=r.cls, ballistic=r.ballistic,
            travel="right" if sgn > 0 else "left", mirrored=bool(sgn < 0),
            crosses_gap=gap, onset_t_s=r.onset_t_s, t_land_s=round(t_land, 2),
            target_x=round(xt, 1) if np.isfinite(xt) else np.nan,
            target_z=round(zt, 1) if np.isfinite(zt) else np.nan,
            paw_x=round(paw[0], 1) if np.isfinite(paw[0]) else np.nan,
            paw_conf=pconf,
            cam2_tracked_pct=round(100 * d.head_raw_ok.mean()),
            peak_head_speed=round(np.nanmax(d.head_speed))))
    meta = pd.DataFrame(meta)
    print(f"{len(W)} jumps loaded")
    print(meta.to_string(index=False))

    def stack(col, key="raw", subset=None):
        ids = [j for j in W if subset is None or j in subset]
        return np.vstack([resample(W[j][key], col, grid) for j in ids]), ids

    by_cls = {c: [j for j in W if W[j]["row"].cls == c] for c in CLASS_COLORS}
    by_cls = {c: v for c, v in by_cls.items() if v}

    fig_speed_all(grid, stack, W)
    fig_speed_epochs(W)
    fig_speed_by_class(grid, stack, by_cls)
    fig_speed_epochs_by_class(W, by_cls)
    fig_geometry(W)
    fig_orientation(grid, stack, by_cls)
    fig_headbody(grid, stack, W)
    fig_posture(grid, stack, by_cls)
    fig_target(grid, W, by_cls, which="target", fname="fig09_target_aiming.png",
               subtitle="Target = the MEASURED position where the head settles after landing.")
    fig_target(grid, W, by_cls, which="edge", fname="fig09b_edge_aiming.png",
               classes=("up->up",),
               subtitle="Target = the LANDING PLATFORM'S FRONT EDGE. up->up only "
                        "(for up->down the edge is behind the animal: mean |error| ~140 deg).")
    fig_target(grid, W, by_cls, which="paw", fname="fig09c_paw_aiming.png",
               subtitle="Target = the MANUALLY ANNOTATED forepaw touchdown point "
                        "(~+-20 px; DLC tracks head only; dotted = low-confidence jump).")
    write_tables(W, meta, grid, stack, by_cls)
    print(f"\nfigures -> {FIG}")


# --------------------------------------------------------------------------- #
# speed
# --------------------------------------------------------------------------- #
def fig_speed_all(grid, stack, W):
    mat, ids = stack("head_speed")
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for row, j in zip(mat, ids):
        ax.plot(grid, row, lw=0.8, alpha=0.55, label=j)
    m, e = np.nanmean(mat, axis=0), sem(mat, axis=0)
    ax.plot(grid, m, lw=3, color="black", label=f"MEAN (n={len(mat)})", zorder=5)
    ax.fill_between(grid, m - e, m + e, color="black", alpha=0.18, zorder=4, linewidth=0)
    ax.axvline(0, color="k", ls="--", lw=1.2)
    for _, a, b in EPOCHS:
        ax.axvspan(a, b, color="gray", alpha=0.05)
    ax.set_xlabel("time from jump onset (s)")
    ax.set_ylabel("head-centroid speed (px/s, Cam2)")
    ax.set_title(f"Head speed, every confirmed jump aligned to onset (n={len(mat)})\n"
                 "onset = frame of peak head speed; shaded band = SEM across jumps")
    ax.legend(fontsize=6, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig01_speed_all_jumps.png"))
    plt.close(fig)


def fig_speed_epochs(W):
    g = np.array([[epoch_means(W[j]["raw"], "head_speed")[n] for n, _, _ in EPOCHS]
                  for j in W])
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    bars(ax, [g], ["all jumps"], ["tab:purple"],
         "Mean head speed by epoch (all confirmed jumps)", "head speed (px/s)")
    ax.set_xlabel("epoch relative to onset")
    fig.text(0.5, 0.005, "error bars = SEM across jumps (SD/sqrt(n)); dots = individual jumps",
             ha="center", fontsize=7, style="italic")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(os.path.join(FIG, "fig02_speed_epochs_bar.png"))
    plt.close(fig)


def fig_speed_by_class(grid, stack, by_cls):
    n = len(by_cls)
    fig, axes = plt.subplots(1, n + 1, figsize=(4.3 * (n + 1), 4.3), sharey=True)
    for ax, (c, ids) in zip(axes, by_cls.items()):
        mat, _ = stack("head_speed", subset=set(ids))
        for row in mat:
            ax.plot(grid, row, lw=0.8, color=CLASS_COLORS[c], alpha=0.35)
        m, e = np.nanmean(mat, axis=0), sem(mat, axis=0)
        ax.plot(grid, m, lw=2.5, color=CLASS_COLORS[c], label=f"mean (n={len(ids)})")
        ax.fill_between(grid, m - e, m + e, color=CLASS_COLORS[c], alpha=0.25, linewidth=0)
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.set_title(f"{c}  (n={len(ids)})")
        ax.set_xlabel("t from onset (s)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("head speed (px/s)")
    for c, ids in by_cls.items():
        mat, _ = stack("head_speed", subset=set(ids))
        axes[-1].plot(grid, np.nanmean(mat, axis=0), lw=2.2,
                      color=CLASS_COLORS[c], label=f"{c} (n={len(ids)})")
    axes[-1].axvline(0, color="k", ls="--", lw=1)
    axes[-1].set_title("class means overlaid")
    axes[-1].set_xlabel("t from onset (s)")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Head speed by jump class, aligned to onset (shaded = SEM across jumps)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig03_speed_by_class.png"))
    plt.close(fig)


def fig_speed_epochs_by_class(W, by_cls):
    groups, labels, colors = [], [], []
    for c, ids in by_cls.items():
        groups.append(np.array([[epoch_means(W[j]["raw"], "head_speed")[n]
                                 for n, _, _ in EPOCHS] for j in ids]))
        labels.append(c)
        colors.append(CLASS_COLORS[c])
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    bars(ax, groups, labels, colors, "Mean head speed by epoch and jump class",
         "head speed (px/s)")
    ax.set_xlabel("epoch relative to onset")
    fig.text(0.5, 0.005, "error bars = SEM across jumps; classes with n=1 have no SEM",
             ha="center", fontsize=7, style="italic")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(os.path.join(FIG, "fig04_speed_epochs_by_class.png"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# geometry validation
# --------------------------------------------------------------------------- #
def fig_geometry(W):
    g = GEOM
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    for ax, use_mirror in zip(axes, [False, True]):
        key = "m" if use_mirror else "raw"
        if not use_mirror:
            ax.add_patch(Rectangle((0, 0), g["left_edge_x"], g["plat_surface_y_up"],
                                   color="tab:gray", alpha=0.18))
            ax.add_patch(Rectangle((g["right_edge_x"], 0), L.IMG - g["right_edge_x"],
                                   g["plat_surface_y_up"], color="tab:gray", alpha=0.18))
            ax.axvline(g["left_edge_x"], color="k", lw=1, ls=":")
            ax.axvline(g["right_edge_x"], color="k", lw=1, ls=":")
            ax.text(g["left_edge_x"] / 2, g["plat_surface_y_up"] + 18, "LEFT platform",
                    ha="center", fontsize=8)
            ax.text((g["right_edge_x"] + L.IMG) / 2, g["plat_surface_y_up"] + 18,
                    "RIGHT platform", ha="center", fontsize=8)
            ax.text((g["left_edge_x"] + g["right_edge_x"]) / 2, g["floor_y_up"] + 25,
                    f"gap {g['right_edge_x'] - g['left_edge_x']:.0f} px",
                    ha="center", fontsize=8, style="italic")
        ax.axhline(g["plat_surface_y_up"], color="k", lw=1.2)
        ax.axhline(g["floor_y_up"], color="saddlebrown", lw=1.2)
        for j, w in W.items():
            d = w[key]
            m = (d.t >= -0.5) & (d.t <= 1.5)
            i0 = int(np.argmin(np.abs(d.t.to_numpy())))
            # mirroring alone leaves left- and right-going jumps in two separate
            # clusters, so also translate each one to put its onset at x = 0
            dx = d.head_x.iloc[i0] if use_mirror else 0.0
            dx = dx if np.isfinite(dx) else 0.0
            ax.plot(d.head_x[m] - dx, d.head_z[m], lw=1.1,
                    color=CLASS_COLORS[w["row"].cls], alpha=0.75)
            ax.plot(d.head_x.iloc[i0] - dx, d.head_z.iloc[i0], "o", ms=5,
                    color=CLASS_COLORS[w["row"].cls], mec="k", mew=0.6, zorder=5)
            # annotated forepaw touchdown point (raw coords left, mirrored right)
            pawx, pawz = w["paw"] if use_mirror else w["paw_raw"]
            if np.isfinite(pawx):
                ax.plot(pawx - dx, pawz, "*", ms=13,
                        color=CLASS_COLORS[w["row"].cls], mec="k", mew=0.7, zorder=6)
        ax.set_xlabel("head x (px)" + ("   [mirrored + centred on onset]" if use_mirror else ""))
        ax.set_ylabel("head height y_up (px)")
        ax.set_title("mirrored to a common travel direction, onset at x=0" if use_mirror
                     else "raw image coordinates + measured arena geometry")
        if use_mirror:
            ax.axvline(0, color="k", ls="--", lw=1)
    handles = [plt.Line2D([], [], color=c, lw=2, label=k) for k, c in CLASS_COLORS.items()]
    handles += [plt.Line2D([], [], color="k", marker="o", ls="", label="onset (t=0)"),
                plt.Line2D([], [], color="k", marker="*", ls="", ms=11, label="forepaw touchdown"),
                plt.Line2D([], [], color="k", lw=1.2, label="platform surface"),
                plt.Line2D([], [], color="saddlebrown", lw=1.2, label="floor")]
    axes[0].legend(handles=handles, fontsize=7, loc="lower left")
    fig.suptitle("Head trajectories -0.5..+1.5 s vs measured arena geometry "
                 "(validation of the platform/floor levels used for classing and targets)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig05_geometry_check.png"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# orientation
# --------------------------------------------------------------------------- #
def fig_orientation(grid, stack, by_cls):
    want = [c for c in ("up->up", "up->down") if c in by_cls]
    rows = [("ang_nape_snout", "nape -> snout  (full head axis)"),
            ("ang_eye_snout", "eye -> snout  (muzzle / pitch proxy)")]
    fig, axes = plt.subplots(len(rows), len(want) + 1,
                             figsize=(4.7 * (len(want) + 1), 4.2 * len(rows)),
                             sharex=True, squeeze=False)
    for ri, (col, nice) in enumerate(rows):
        for ci, c in enumerate(want):
            ax = axes[ri][ci]
            mat, _ = stack(col, key="m", subset=set(by_cls[c]))
            for r in mat:
                ax.plot(grid, r, lw=0.8, alpha=0.45, color=CLASS_COLORS[c])
            mu, se = L.circmean_deg(mat, axis=0), L.circsem_deg(mat, axis=0)
            ax.plot(grid, mu, lw=2.6, color="black", label=f"circular mean (n={len(mat)})")
            ax.fill_between(grid, mu - se, mu + se, color="black", alpha=0.2, linewidth=0)
            # how many jumps actually contribute at each time point
            nn = np.sum(np.isfinite(mat), axis=0)
            ax2 = ax.twinx()
            ax2.fill_between(grid, 0, nn, color="tab:orange", alpha=0.12, linewidth=0)
            ax2.set_ylim(0, len(mat) * 4)
            ax2.set_yticks([0, len(mat)])
            ax2.set_ylabel("n jumps", fontsize=7, color="tab:orange")
            ax2.tick_params(labelsize=6, colors="tab:orange")
            ax2.grid(False)
            ax.axvline(0, color="k", ls="--", lw=1)
            ax.axhline(0, color="tab:gray", lw=0.8)
            ax.set_ylim(-180, 90)
            ax.set_title(f"{c}\n{nice}", fontsize=9)
            ax.legend(fontsize=7, loc="lower left")
            if ci == 0:
                ax.set_ylabel("angle from +x (deg)\n0 = travel dir, +90 = nose up")
        ax = axes[ri][-1]
        for c in want:
            mat, _ = stack(col, key="m", subset=set(by_cls[c]))
            ax.plot(grid, L.circmean_deg(mat, axis=0), lw=2.2,
                    color=CLASS_COLORS[c], label=f"{c} (n={len(by_cls[c])})")
        ax.set_ylim(-180, 90)
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.axhline(0, color="tab:gray", lw=0.8)
        ax.set_title("class means overlaid", fontsize=9)
        ax.legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("time from jump onset (s)")
    fig.suptitle("Head orientation relative to the travel (+x) axis\n"
                 "mirrored so every jump travels frame-right; +90 deg = nose up, "
                 "-90 deg = nose down")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig06_head_orientation.png"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# head vs body
# --------------------------------------------------------------------------- #
def fig_headbody(grid, stack, W):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    hs, _ = stack("head_speed")
    bs, _ = stack("body_speed")
    rs, _ = stack("rel_speed")
    for mat, name, col in [(hs, "head (DLC centroid)", "tab:blue"),
                           (bs, "body (blob centroid)", "tab:orange")]:
        m, e = np.nanmean(mat, axis=0), sem(mat, axis=0)
        axes[0].plot(grid, m, lw=2.2, color=col, label=name)
        axes[0].fill_between(grid, m - e, m + e, color=col, alpha=0.22, linewidth=0)
    axes[0].axvline(0, color="k", ls="--", lw=1)
    axes[0].legend(fontsize=8)
    axes[0].set_title("Head vs body speed (all jumps)")
    axes[0].set_ylabel("speed (px/s)")

    m, e = np.nanmean(rs, axis=0), sem(rs, axis=0)
    axes[1].plot(grid, m, lw=2.2, color="tab:purple")
    axes[1].fill_between(grid, m - e, m + e, color="tab:purple", alpha=0.25, linewidth=0)
    axes[1].axhline(0, color="k", lw=0.9)
    axes[1].axvline(0, color="k", ls="--", lw=1)
    axes[1].set_title("head speed - body speed\n(+ = head outrunning the body)")
    axes[1].set_ylabel("px/s")

    lags = np.arange(-50, 51)
    cc = []
    for h, b in zip(hs, bs):
        ok = np.isfinite(h) & np.isfinite(b)
        if ok.sum() < 120:
            continue
        h0 = (h[ok] - h[ok].mean()) / (h[ok].std() + 1e-9)
        b0 = (b[ok] - b[ok].mean()) / (b[ok].std() + 1e-9)
        c = np.correlate(h0, b0, "full") / len(h0)
        mid = len(c) // 2
        cc.append(c[mid - 50:mid + 51])
    if cc:
        cc = np.vstack(cc)
        mm, ee = np.nanmean(cc, axis=0), sem(cc, axis=0)
        axes[2].plot(lags * 10, mm, lw=2.2, color="tab:green")
        axes[2].fill_between(lags * 10, mm - ee, mm + ee, color="tab:green",
                             alpha=0.25, linewidth=0)
        pk = lags[int(np.nanargmax(mm))] * 10
        axes[2].axvline(pk, color="tab:red", ls=":", lw=1.5, label=f"peak at {pk:+.0f} ms")
        axes[2].axvline(0, color="k", lw=0.9)
        axes[2].legend(fontsize=8)
    axes[2].set_title("head-body speed cross-correlation\n(negative lag = head leads body)")
    axes[2].set_xlabel("lag (ms)")
    axes[2].set_ylabel("normalised correlation")
    for ax in axes[:2]:
        ax.set_xlabel("time from jump onset (s)")

    # body peak lag -- the part of this comparison that is NOT circular
    lags = []
    for b in bs:
        if np.isfinite(b).sum() > 50:
            lags.append(grid[int(np.nanargmax(np.where(np.isfinite(b), b, -np.inf)))])
    if lags:
        axes[0].annotate(f"body peak lags head by\nmedian {np.median(lags)*1000:+.0f} ms "
                         f"(n={len(lags)})", xy=(0.55, 0.72), xycoords="axes fraction",
                         fontsize=8, bbox=dict(fc="white", ec="gray", alpha=0.85))
    fig.suptitle("Head vs body kinematics. Body = proxy blob centroid (128 px grid, "
                 "upscaled x8) -- a coarse trend signal, not a precise one.\n"
                 "CAVEAT: t=0 is DEFINED as the head-speed peak, so the head peaking at 0 "
                 "is circular. The body's lag and the post-onset sign flip are not.",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig07_head_vs_body.png"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# posture
# --------------------------------------------------------------------------- #
def fig_posture(grid, stack, by_cls):
    pairs = [("d_nape_snout", "nape-snout (full head axis)"),
             ("d_nape_eye", "nape-eye"),
             ("d_eye_snout", "eye-snout (muzzle)"),
             ("d_nape_head_top", "nape-head_top")]
    fig, axes = plt.subplots(1, len(pairs), figsize=(3.7 * len(pairs), 4.2), sharex=True)
    for ax, (col, nice) in zip(axes, pairs):
        for c, ids in by_cls.items():
            mat, _ = stack(col, subset=set(ids))
            if not np.isfinite(mat).any():
                continue
            m, e = np.nanmean(mat, axis=0), sem(mat, axis=0)
            ax.plot(grid, m, lw=2, color=CLASS_COLORS[c], label=f"{c} (n={len(ids)})")
            ax.fill_between(grid, m - e, m + e, color=CLASS_COLORS[c],
                            alpha=0.2, linewidth=0)
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.set_title(nice, fontsize=9)
        ax.set_xlabel("t from onset (s)")
    axes[0].set_ylabel("projected distance (px)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Inter-keypoint distances in the Cam2 image plane. These SHORTEN when the "
                 "head axis rotates out of the image plane\n(yaw toward/away from the camera), "
                 "so a dip = head turned off the sagittal plane, not a shrinking head.")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig08_posture_distances.png"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# target aiming
# --------------------------------------------------------------------------- #
TARGET_MIN_DIST = 80.0      # px; closer than this the bearing is meaningless


def target_error(w, grid, which="target"):
    """Signed angle between the head axis and the head->target vector.

    which='target' -> the measured landing position of the head
    which='edge'   -> the landing platform's front edge (gap-crossing jumps only)

    Blanked once the animal has essentially ARRIVED: within TARGET_MIN_DIST of
    the target, or more than 0.15 s past landing. The bearing to a target you
    are standing on is a 0/0 -- it flips through +-180 deg on tracking noise and
    would otherwise dominate the pooled means.
    """
    d = w["m"]
    tx, tz = w[which]
    if not np.isfinite(tx):
        return np.full(len(grid), np.nan)
    dx = (tx - d.head_x).to_numpy()
    dz = (tz - d.head_z).to_numpy()
    dist = np.hypot(dx, dz)
    err = L.wrap180(d.ang_nape_snout.to_numpy() - L.ang_deg(dx, dz))
    keep = (dist >= TARGET_MIN_DIST) & (d.t.to_numpy() <= w["t_land"] + 0.15)
    err = np.where(keep, err, np.nan)
    return resample(d[["t"]].assign(e=err), "e", grid)


TGT_XLIM = (-1.0, 1.2)          # aiming is about approach + jump, not the aftermath
MIN_N_POOL = 3                  # blank the pooled mean where fewer jumps contribute


def fig_target(grid, W, by_cls, which="target", fname="fig09_target_aiming.png",
               subtitle="Target = the measured position where the head settles after landing.",
               classes=("up->up", "up->down")):
    want = [c for c in classes if c in by_cls]
    pool = {c: [j for j in by_cls[c] if np.isfinite(W[j][which][0])] for c in want}
    want = [c for c in want if pool[c]]
    if not want:
        print(f"  (no jumps with a valid '{which}' target -- skipping {fname})")
        return
    disp = (grid >= TGT_XLIM[0]) & (grid <= TGT_XLIM[1])
    fig = plt.figure(figsize=(14.5, 8.6))
    gs = fig.add_gridspec(2, len(want) + 1, height_ratios=[1, 1])

    per_class = {}
    for ci, c in enumerate(want):
        ids = pool[c]
        mat = np.vstack([target_error(W[j], grid, which) for j in ids])
        confs = [W[j].get("paw_conf", "") if which == "paw" else "" for j in ids]
        per_class[c] = mat
        ax = fig.add_subplot(gs[0, ci])
        for r, j, cf in zip(mat, ids, confs):
            style = dict(lw=0.8, alpha=0.45) if cf != "low" else dict(lw=0.8, alpha=0.35, ls=":")
            ax.plot(grid[disp], r[disp], color=CLASS_COLORS[c], **style)
        nn = np.sum(np.isfinite(mat), axis=0)
        mu = L.circmean_deg(mat, axis=0)
        se = L.circsem_deg(mat, axis=0)
        mu = np.where(nn >= MIN_N_POOL, mu, np.nan)
        se = np.where(nn >= MIN_N_POOL, se, np.nan)
        ax.plot(grid[disp], mu[disp], lw=2.6, color="black",
                label=f"circular mean (n={len(mat)})")
        ax.fill_between(grid[disp], (mu - se)[disp], (mu + se)[disp],
                        color="black", alpha=0.2, linewidth=0)
        ax.axhline(0, color="tab:green", lw=1.4)
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.set_xlim(*TGT_XLIM)
        ax.set_ylim(-140, 160)
        ax.set_title(f"{c}  (n={len(ids)})", fontsize=9)
        ax.set_ylabel("head-axis error vs target (deg)\n0 = pointing straight at it")
        ax.legend(fontsize=7)

        axh = fig.add_subplot(gs[1, ci])
        edges_t = np.arange(TGT_XLIM[0], TGT_XLIM[1] + 1e-6, 0.05)
        edges_a = np.arange(-140, 161, 10)
        H = np.full((len(edges_a) - 1, len(edges_t) - 1), np.nan)
        for ti in range(len(edges_t) - 1):
            m = (grid >= edges_t[ti]) & (grid < edges_t[ti + 1])
            vals = mat[:, m].ravel()
            vals = vals[np.isfinite(vals)]
            if len(vals) >= 3:
                h, _ = np.histogram(vals, bins=edges_a)
                H[:, ti] = h / h.sum()
        im = axh.pcolormesh(edges_t, edges_a, np.ma.masked_invalid(H),
                            cmap="magma", shading="auto")
        axh.axhline(0, color="lime", lw=1.2)
        axh.axvline(0, color="white", ls="--", lw=1)
        axh.set_xlabel("time from jump onset (s)")
        axh.set_ylabel("aiming error (deg)")
        axh.set_title(f"{c}: aiming-error density (column-normalised)", fontsize=9)
        axh.grid(False)
        fig.colorbar(im, ax=axh, fraction=0.03, label="fraction of frames")

    ax = fig.add_subplot(gs[0, -1])
    for c in want:
        mat = per_class[c]
        nn = np.sum(np.isfinite(mat), axis=0)
        y = np.where(nn >= MIN_N_POOL, np.nanmean(np.abs(mat), axis=0), np.nan)
        ax.plot(grid[disp], y[disp], lw=2.2, color=CLASS_COLORS[c],
                label=f"{c} (n={len(pool[c])})")
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_xlim(*TGT_XLIM)
    ax.set_title("mean |aiming error|  (lower = better aimed)", fontsize=9)
    ax.set_ylabel("deg")
    ax.legend(fontsize=7)

    axb = fig.add_subplot(gs[1, -1])
    data, labels, colors = [], [], []
    for c in want:
        mat = per_class[c]
        pre = np.abs(mat[:, (grid >= -0.6) & (grid < -0.1)])
        dur = np.abs(mat[:, (grid >= -0.05) & (grid < 0.15)])
        for arr, tag in [(pre, "approach\n[-0.6,-0.1)s"), (dur, "at onset\n[-0.05,0.15)s")]:
            v = np.nanmean(arr, axis=1)
            data.append(v[np.isfinite(v)])
            labels.append(f"{c}\n{tag}")
            colors.append(CLASS_COLORS[c])
    bp = axb.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.55)
    for i, v in enumerate(data, 1):
        axb.plot(np.full(len(v), i) + np.random.default_rng(1).uniform(-.08, .08, len(v)),
                 v, "o", ms=3.5, color="black", alpha=0.6)
    axb.set_ylabel("|aiming error| (deg), per-jump mean")
    axb.set_title("approach vs onset aiming", fontsize=9)
    axb.tick_params(axis="x", labelsize=6.5)

    fig.suptitle("Head aiming: angle between the nape->snout head axis and the "
                 "direction to the target.\n0 deg = head pointing straight at it.  "
                 + subtitle)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)


# --------------------------------------------------------------------------- #
def write_tables(W, meta, grid, stack, by_cls):
    rows = []
    for j, w in W.items():
        d, r = w["raw"], w["row"]
        rec = dict(jump=j, src_clips=r.src_clips, cls=r.cls, ballistic=r.ballistic,
                   travel="right" if w["sgn"] > 0 else "left")
        for col, tag in [("head_speed", "hspd"), ("body_speed", "bspd"),
                         ("rel_speed", "relspd"), ("d_nape_snout", "d_nape_snout")]:
            if col in d:
                for n, v in epoch_means(d, col).items():
                    rec[f"{tag}_{n}"] = round(v, 1) if np.isfinite(v) else np.nan
        for which, tag in [("target", "aimerr"), ("edge", "edgeerr"), ("paw", "pawerr")]:
            err = target_error(w, grid, which)
            for n, a, b in EPOCHS:
                v = np.nanmean(np.abs(err[(grid >= a) & (grid < b)]))
                rec[f"{tag}_{n}"] = round(v, 1) if np.isfinite(v) else np.nan
        rec["paw_conf"] = w.get("paw_conf", "")
        rows.append(rec)
    summ = meta.merge(pd.DataFrame(rows), on=["jump", "cls", "ballistic", "travel"],
                      how="left", suffixes=("", "_y"))
    summ.to_csv(os.path.join(L.V3, "v3_summary.csv"), index=False)

    stats = []
    for label, ids in [("ALL", list(W))] + [(c, v) for c, v in by_cls.items()]:
        for col in ["head_speed", "body_speed", "rel_speed"]:
            g = np.array([[epoch_means(W[j]["raw"], col)[n] for n, _, _ in EPOCHS]
                          for j in ids])
            for k, (n, _, _) in enumerate(EPOCHS):
                v = g[:, k]
                v = v[np.isfinite(v)]
                stats.append(dict(group=label, metric=col, epoch=n, n=len(v),
                                  mean=round(float(np.mean(v)), 2) if len(v) else np.nan,
                                  sd=round(float(np.std(v, ddof=1)), 2) if len(v) > 1 else np.nan,
                                  sem=round(float(np.std(v, ddof=1) / np.sqrt(len(v))), 2)
                                  if len(v) > 1 else np.nan))
        # angles need circular statistics, and come from the mirrored frame
        for col in ["ang_nape_snout", "ang_eye_snout"]:
            for n, a, b in EPOCHS:
                v = []
                for j in ids:
                    d = W[j]["m"]
                    m = (d.t >= a) & (d.t < b)
                    v.append(L.circmean_deg(d[col][m].to_numpy()))
                v = np.array(v, float)
                v = v[np.isfinite(v)]
                stats.append(dict(group=label, metric=col + " (mirrored, circular)",
                                  epoch=n, n=len(v),
                                  mean=round(float(L.circmean_deg(v)), 2) if len(v) else np.nan,
                                  sd=round(float(np.std(v, ddof=1)), 2) if len(v) > 1 else np.nan,
                                  sem=round(float(np.std(v, ddof=1) / np.sqrt(len(v))), 2)
                                  if len(v) > 1 else np.nan))
    pd.DataFrame(stats).to_csv(os.path.join(L.V3, "v3_epoch_stats.csv"), index=False)
    print(f"\nwrote v3_summary.csv and v3_epoch_stats.csv")


if __name__ == "__main__":
    main()
