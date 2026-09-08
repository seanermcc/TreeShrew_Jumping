"""
Head-axis stability analyses (v3).

Adds three things that were missing from v3_analysis / v3_extra:

  fig15  WITHIN-TRIAL head-axis jitter, per jump, epoch by epoch.
         Every earlier orientation number (fig06, v3_epoch_stats.csv) is an SD
         ACROSS jumps at a fixed time - it says nothing about how still one
         animal holds its head within a single approach. Here each jump gives
         one number per epoch:
           - circular SD of the head-axis angle (raw and linear-detrended)
           - a robust peak |angular velocity| (95th pct of |d(angle)/dt|)
           - the 5-95 percentile RANGE of the angle over the approach
           - the 5-95 range of the head-axis -> target BEARING over the approach
         and the figure is approach-vs-jump-vs-post box/strip plots of those.

  fig16  FIXATION POINT / retinal-slip. Project the nape->snout head axis
         forward to a target plane (vertical plane at the landing-platform edge
         for gap crossings; horizontal floor plane for drop landings) and track
         the intersection pixel frame by frame. If the animal fixates the
         landing zone the intersection sits still during the approach and only
         scatters once the jump/land/right sequence starts.

Run after v3_analysis.py (needs v3/v3_jumps.csv and the keypoint cache).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import v3_lib as L
import v3_analysis as A
import v3_extra as X
from v3_build import GEOM

FIG = A.FIG
EP = [("approach", -1.0, 0.0), ("jump", 0.0, 0.2), ("post", 0.2, 3.0)]
EP_COL = {"approach": "tab:blue", "jump": "tab:orange", "post": "tab:green"}
APPROACH = (-1.0, -0.15)          # the window the "is the head still" claim is about
                                  # (ends before the takeoff ramp, which starts ~-0.15 s)


# --------------------------------------------------------------------------- #
# small stats helpers
# --------------------------------------------------------------------------- #
def circ_sd_deg(a):
    """SD of an angle sample, robust to the (irrelevant here) wrap point."""
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if len(a) < 3:
        return np.nan
    m = L.circmean_deg(a)
    return float(np.std(L.wrap180(a - m), ddof=1))


def detrended_sd_deg(t, a):
    """SD of the angle after removing a linear drift - separates 'noisy' from
    'slowly turning'. These angles never wrap, so a plain polyfit is fine."""
    t, a = np.asarray(t, float), np.asarray(a, float)
    ok = np.isfinite(t) & np.isfinite(a)
    if ok.sum() < 5:
        return np.nan
    p = np.polyfit(t[ok], a[ok], 1)
    return float(np.std(a[ok] - np.polyval(p, t[ok]), ddof=1))


def peak_ang_vel(t, a, pct=95):
    """Robust peak |angular velocity| in deg/s: the `pct`-th percentile of
    |d(angle)/dt| over consecutive tracked frames (ignores 1-frame glitches)."""
    t, a = np.asarray(t, float), np.asarray(a, float)
    ok = np.isfinite(t) & np.isfinite(a)
    t, a = t[ok], a[ok]
    if len(t) < 4:
        return np.nan
    dt = np.diff(t)
    v = np.abs(np.diff(a)) / dt
    v = v[(dt > 0) & (dt < 0.05)]              # only across genuinely adjacent frames
    return float(np.nanpercentile(v, pct)) if len(v) else np.nan


def pctl_range(a, lo=5, hi=95):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if len(a) < 3:
        return np.nan
    return float(np.nanpercentile(a, hi) - np.nanpercentile(a, lo))


def safe_csv(df, path):
    """Write a CSV, but don't abort the whole figure run if it's open in Excel."""
    try:
        df.to_csv(path, index=False)
    except PermissionError:
        print(f"  !! {os.path.basename(path)} is locked (open elsewhere?) - not rewritten")


def seg(d, a, b, col):
    m = (d.t >= a) & (d.t < b)
    return d.t.to_numpy()[m], pd.to_numeric(d[col], errors="coerce").to_numpy()[m]


# --------------------------------------------------------------------------- #
# fixation-point projection
# --------------------------------------------------------------------------- #
MAX_REACH_PX = 2500.0             # blank grazing / away-pointing projections
MIN_COS = 0.12                    # ray too parallel to the plane -> unstable


def fixation_plane(w):
    """(kind, value, label) for the plane this jump's head axis is projected to.

    up->down : horizontal plane at the measured landing height  -> track x
    else     : vertical plane at the landing-platform front edge -> track height
    """
    r = w["row"]
    if r.cls == "up->down":
        _, zt = w["target"]
        return ("z", zt, "fixation x on the floor plane (px)")
    xe = w["edge"][0] if w["gap"] and np.isfinite(w["edge"][0]) else w["target"][0]
    return ("x", xe, "fixation height on the edge plane (px)")


def project_axis(d, plane):
    kind, val, _ = plane
    if not np.isfinite(val):
        return np.full(len(d), np.nan)
    hx, hz = d.head_x.to_numpy(), d.head_z.to_numpy()
    th = np.deg2rad(pd.to_numeric(d.ang_nape_snout, errors="coerce").to_numpy())
    cx, sz = np.cos(th), np.sin(th)
    with np.errstate(invalid="ignore", divide="ignore"):
        if kind == "x":
            s = (val - hx) / cx
            hit = hz + s * sz
            bad = ~np.isfinite(s) | (s <= 0) | (np.abs(cx) < MIN_COS)
        else:
            s = (val - hz) / sz
            hit = hx + s * cx
            bad = ~np.isfinite(s) | (s <= 0) | (np.abs(sz) < MIN_COS)
        bad = bad | (np.abs(s) > MAX_REACH_PX)
    return np.where(bad, np.nan, hit)


# --------------------------------------------------------------------------- #
def per_jump_table(W):
    rows = []
    for j, w in W.items():
        d = w["m"]
        r = w["row"]
        # target for the bearing-jitter column: edge for gap crossings, else the
        # measured landing head position
        if w["gap"] and np.isfinite(w["edge"][0]):
            tx, tz = w["edge"]
        else:
            tx, tz = w["target"]
        rec = dict(jump=j, cls=r.cls, ballistic=str(r.ballistic or "?"),
                   tracked_pct=r.cam2_tracked_pct if hasattr(r, "cam2_tracked_pct") else np.nan)
        for name, a, b in EP:
            t, ang = seg(d, a, b, "ang_nape_snout")
            rec[f"sd_{name}"] = circ_sd_deg(ang)
            rec[f"sddet_{name}"] = detrended_sd_deg(t, ang)
            rec[f"pkav_{name}"] = peak_ang_vel(t, ang)
        # approach-only ranges
        ta, aa = seg(d, *APPROACH, "ang_nape_snout")
        rec["ang_range_approach"] = pctl_range(aa)
        if np.isfinite(tx):
            dx = tx - d.head_x.to_numpy()
            dz = tz - d.head_z.to_numpy()
            bearing_err = L.wrap180(pd.to_numeric(d.ang_nape_snout, errors="coerce").to_numpy()
                                    - L.ang_deg(dx, dz))
            near = np.hypot(dx, dz) < A.TARGET_MIN_DIST
            bearing_err = np.where(near, np.nan, bearing_err)
            mA = (d.t.to_numpy() >= APPROACH[0]) & (d.t.to_numpy() < APPROACH[1])
            rec["bearing_sd_approach"] = circ_sd_deg(bearing_err[mA])
            rec["bearing_range_approach"] = pctl_range(bearing_err[mA])
        else:
            rec["bearing_sd_approach"] = np.nan
            rec["bearing_range_approach"] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def _strip_box(ax, data, labels, colors, ylab, title, logy=False):
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.55,
                    showfliers=False, medianprops=dict(color="black"))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.35)
    rng = np.random.default_rng(3)
    for i, (v, c) in enumerate(zip(data, colors), 1):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        ax.plot(np.full(len(v), i) + rng.uniform(-0.08, 0.08, len(v)), v, "o",
                ms=4, color=c, mec="k", mew=0.4, alpha=0.9)
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=9)


def _pick_example(W):
    """Up->up gap crossing with the best APPROACH-phase nape+snout coverage - used
    for the schematics and the fig16b worked example (which draws real frames, so
    it needs the keypoints actually present before onset). Rightward jumps break
    ties: they need no mirroring, so the drawn geometry is the raw geometry."""
    def score(j):
        d = W[j]["raw"]
        m = (d.t.to_numpy() >= APPROACH[0]) & (d.t.to_numpy() < APPROACH[1])
        cov = (np.isfinite(d.nape_x.to_numpy()[m]) & np.isfinite(d.snout_x.to_numpy()[m])).mean()
        return cov + 0.05 * (W[j]["sgn"] > 0)
    cand = [j for j, w in W.items()
            if w["row"].cls == "up->up" and w["gap"] and np.isfinite(w["edge"][0])]
    return max(cand or list(W), key=score)


def _angle_schematic(ax):
    """What 'head-axis angle' means: direction of the nape->snout vector."""
    ax.set_aspect("equal")
    nape, snout = np.array([0, 0]), np.array([np.cos(np.deg2rad(-35)),
                                              np.sin(np.deg2rad(-35))])
    ax.annotate("", xy=(1.35, 0), xytext=(-0.35, 0),
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.text(1.36, 0.03, "+x  (image-right =\ntravel direction)", fontsize=7, color="gray")
    ax.annotate("", xy=snout * 1.15, xytext=nape,
                arrowprops=dict(arrowstyle="-|>", color="tab:blue", lw=2))
    th = np.linspace(0, np.deg2rad(-35), 30)
    ax.plot(0.55 * np.cos(th), 0.55 * np.sin(th), color="tab:blue", lw=1)
    ax.text(0.62, -0.22, "head-axis\nangle", fontsize=8, color="tab:blue")
    ax.plot(*nape, "o", ms=9, color="magenta"); ax.text(-0.02, 0.08, "nape", fontsize=8)
    ax.plot(*snout, "o", ms=9, color="red"); ax.text(snout[0] + .03, snout[1], "snout", fontsize=8)
    ax.set_xlim(-0.5, 2.3); ax.set_ylim(-0.9, 0.7)
    ax.axis("off")
    ax.set_title("head-axis angle = atan2( snout_z-nape_z , snout_x-nape_x )\n"
                 "0 deg = along travel,  +90 = nose up,  -90 = nose down", fontsize=8)


def _circ_sd_schematic(ax, approach_deg, post_deg):
    """What circular SD is: spread of a set of directions about their mean.
    Two mini-dials from the same example jump - approach (steady) vs post (not)."""
    ax.set_aspect("equal")
    circ = np.linspace(0, 2 * np.pi, 200)
    for cx, sample, lab in [(-1.35, approach_deg, "approach"), (1.35, post_deg, "post")]:
        a = np.deg2rad(np.asarray(sample, float))
        a = a[np.isfinite(a)]
        if len(a) < 3:
            continue
        ax.plot(cx + np.cos(circ), np.sin(circ), color="0.8", lw=1)
        for ang in a:
            ax.plot([cx, cx + np.cos(ang)], [0, np.sin(ang)],
                    color="tab:blue", lw=0.7, alpha=0.35)
        mean = np.arctan2(np.nanmean(np.sin(a)), np.nanmean(np.cos(a)))
        ax.annotate("", xy=(cx + np.cos(mean), np.sin(mean)), xytext=(cx, 0),
                    arrowprops=dict(arrowstyle="-|>", color="k", lw=2))
        sd = float(np.std(L.wrap180(np.degrees(a) - np.degrees(mean)), ddof=1))
        ax.text(cx, -1.45, f"{lab}\ncirc SD = {sd:.1f} deg", ha="center", fontsize=8)
    ax.set_xlim(-2.7, 2.7); ax.set_ylim(-1.9, 1.4)
    ax.axis("off")
    ax.set_title("circular SD = SD of (each frame's angle - the circular-mean angle),\n"
                 "wrapped to +-180 deg  (so 179 and -179 count as 2 deg apart, not 358).\n"
                 "same example jump, both windows:", fontsize=8)


def fig_stability(df, W):
    up = df[df.cls == "up->up"]
    ex = _pick_example(W)
    de = W[ex]["raw"]
    _tt = de.t.to_numpy()
    ex_ang = de.ang_nape_snout.to_numpy()[(_tt >= APPROACH[0]) & (_tt < APPROACH[1])]
    ex_ang_post = de.ang_nape_snout.to_numpy()[(_tt >= 0.2) & (_tt < 3.0)]
    fig, axes = plt.subplots(3, 3, figsize=(15, 12.5))

    _strip_box(axes[0, 0],
               [df[f"sd_{n}"].to_numpy() for n, _, _ in EP],
               [n for n, _, _ in EP], [EP_COL[n] for n, _, _ in EP],
               "circular SD of head-axis angle (deg)",
               f"Within-trial head-axis jitter  (all {len(df)} jumps)")
    _strip_box(axes[0, 1],
               [df[f"sddet_{n}"].to_numpy() for n, _, _ in EP],
               [n for n, _, _ in EP], [EP_COL[n] for n, _, _ in EP],
               "detrended SD of head-axis angle (deg)",
               "Same, linear drift removed  (pure noise/tremor)")
    _strip_box(axes[0, 2],
               [df[f"pkav_{n}"].to_numpy() for n, _, _ in EP],
               [n for n, _, _ in EP], [EP_COL[n] for n, _, _ in EP],
               "robust peak |angular velocity| (deg/s, 95th pct)",
               "How fast the head axis rotates, per epoch", logy=True)

    _strip_box(axes[1, 0],
               [up["ang_range_approach"].to_numpy(), up["bearing_range_approach"].to_numpy()],
               ["head-axis\nangle", "axis->target\nbearing error"],
               ["tab:blue", "tab:purple"],
               "5-95 percentile range over the approach (deg)",
               "Approach spread, up->up  (target = platform edge)")

    # approach vs jump SD, paired, per jump
    ax = axes[1, 1]
    for r in df.itertuples():
        c = A.CLASS_COLORS.get(r.cls, "gray")
        ax.plot([0, 1, 2], [r.sd_approach, r.sd_jump, r.sd_post], "-o", ms=4,
                color=c, alpha=0.7, lw=1)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["approach", "jump", "post"])
    ax.set_ylabel("circular SD of head-axis angle (deg)")
    ax.set_title("Paired per jump: steady through approach + jump,\n"
                 "then destabilises after landing", fontsize=9)
    for c, col in A.CLASS_COLORS.items():
        if (df.cls == c).any():
            ax.plot([], [], "-o", color=col, label=c)
    ax.legend(fontsize=7)

    # summary text
    ax = axes[1, 2]
    ax.axis("off")
    lines = ["MEDIAN (IQR) across jumps", ""]
    for n, _, _ in EP:
        v = df[f"sd_{n}"].dropna()
        vd = df[f"sddet_{n}"].dropna()
        pk = df[f"pkav_{n}"].dropna()
        lines.append(f"{n:>9}:  SD {v.median():5.1f} deg  "
                     f"(detr {vd.median():4.1f})   "
                     f"peak |w| {pk.median():6.0f} deg/s")
    lines += ["", "up->up approach only:",
              f"  head-axis angle 5-95 range : {up['ang_range_approach'].median():.1f} deg",
              f"  axis->edge bearing 5-95    : {up['bearing_range_approach'].median():.1f} deg",
              f"  axis->edge bearing SD      : {up['bearing_sd_approach'].median():.1f} deg"]
    n_lo = int((df[["sd_approach", "sd_jump"]].max(axis=1) < df.sd_post).sum())
    lines += ["", f"{n_lo}/{df.sd_post.notna().sum()} jumps: approach & jump SD",
              "  both below the post SD"]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace",
            fontsize=9, transform=ax.transAxes)

    # bottom row: how the two quantities on the y-axes are defined
    _angle_schematic(axes[2, 0])
    _circ_sd_schematic(axes[2, 1], ex_ang, ex_ang_post)
    ax = axes[2, 2]
    ax.axis("off")
    ax.text(0.0, 1.0,
            "HOW TO READ THIS FIGURE\n\n"
            "x-axis (top 3 panels): the 3 time windows,\n"
            "relative to jump onset t=0 (= peak head speed):\n"
            "   approach [-1.0, -0.15) s   before the takeoff ramp\n"
            "   jump     [0.0,  0.20) s   the launch itself\n"
            "   post     [0.2,  3.0)  s   flight + land + right\n\n"
            "y-axes, all computed PER JUMP then plotted as one\n"
            "dot per jump (box = median / IQR across jumps):\n"
            "  - circ SD of head-axis angle: how much the\n"
            "    nape->snout direction wobbles within that\n"
            "    window of that one trial (see bottom-left).\n"
            "  - detrended SD: same after subtracting a straight\n"
            "    line - removes slow turning, leaves tremor/noise.\n"
            "  - peak |angular velocity|: 95th percentile of\n"
            "    |change in angle| / |change in time| between\n"
            "    adjacent tracked frames = fastest re-aim.\n"
            "  - 5-95 range: 95th minus 5th percentile of the\n"
            "    angle (or of the axis->edge bearing error)\n"
            "    over the approach.\n\n"
            f"example jump for the schematics: {ex}",
            va="top", ha="left", family="monospace", fontsize=8, transform=ax.transAxes)

    fig.suptitle("fig15  Within-trial head-axis stability (nape->snout axis, "
                 "mirrored so travel = +x).\n"
                 "Each point = ONE jump. Contrast with fig06 / v3_epoch_stats.csv "
                 "which are SDs ACROSS jumps.", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig15_head_axis_stability.png"))
    plt.close(fig)


def fig_fixation(W, grid):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9),
                             gridspec_kw=dict(height_ratios=[1.4, 1]))
    axT = axes[0, 0]
    axT2 = axes[0, 1]
    per = []
    for j, w in W.items():
        plane = fixation_plane(w)
        hit = project_axis(w["m"], plane)
        d = w["m"]
        t = d.t.to_numpy()
        # centre on the approach median so jumps overlay
        mA = (t >= APPROACH[0]) & (t < APPROACH[1]) & np.isfinite(hit)
        if mA.sum() < 5:
            continue
        base = np.nanmedian(hit[mA])
        cc = A.CLASS_COLORS.get(w["row"].cls, "gray")
        ax = axT if plane[0] == "x" else axT2
        ax.plot(t, hit - base, lw=1, color=cc, alpha=0.55)
        # dispersion metrics
        def rng(a, b):
            m = (t >= a) & (t < b) & np.isfinite(hit)
            return pctl_range(hit[m]) if m.sum() >= 4 else np.nan
        per.append(dict(jump=j, cls=w["row"].cls, plane=plane[0],
                        approach=rng(*APPROACH), jump_ep=rng(0.0, 0.2),
                        post=rng(0.2, 1.0),
                        approach_sd=float(np.nanstd(hit[mA]))))
    for ax, ttl in [(axT, "gap crossings - projected onto the platform-edge plane"),
                    (axT2, "drop landings - projected onto the floor plane")]:
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_xlim(-1.0, 1.0)
        ax.set_xlabel("time from jump onset (s)")
        ax.set_ylabel("fixation point on the plane,\ncentred on approach median (px)")
        ax.set_title(ttl, fontsize=9)
        ax.set_ylim(-250, 250)

    pf = pd.DataFrame(per)
    ax = axes[1, 0]
    if len(pf):
        _strip_box(ax, [pf.approach.to_numpy(), pf.jump_ep.to_numpy(), pf.post.to_numpy()],
                   [f"approach\n[{APPROACH[0]:.0f},{APPROACH[1]:.2f})s",
                    "jump\n[0,0.2)s", "post\n[0.2,1)s"],
                   ["tab:blue", "tab:orange", "tab:green"],
                   "5-95 range of the fixation point (px)",
                   "Retinal slip: fixation-point spread by epoch", logy=True)
    ax = axes[1, 1]
    ax.axis("off")
    if len(pf):
        lines = ["fixation-point 5-95 range (px), median across jumps", ""]
        for e in ("approach", "jump_ep", "post"):
            lines.append(f"  {e:>9}: {pf[e].median():6.1f}")
        lines += ["", f"n jumps with a usable projection: {len(pf)}",
                  "",
                  "approach-phase SD, per jump:"]
        for r in pf.sort_values("approach_sd").itertuples():
            lines.append(f"  {r.jump}  {r.cls:8s}  {r.approach_sd:6.1f} px")
        ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left",
                family="monospace", fontsize=8.5, transform=ax.transAxes)

    fig.suptitle("fig16  Fixation point / retinal slip. Where the nape->snout "
                 "head axis, extended forward, crosses the landing plane.\n"
                 "Flat during the approach = the animal holds gaze on the landing "
                 "zone; it scatters only once the jump starts.", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig16_fixation_point.png"))
    plt.close(fig)
    return pf


# --------------------------------------------------------------------------- #
def fig_fixation_worked_example(W, ex):
    """fig14-style: the fig16 projection drawn on real Cam2 frames for one jump.

    For each frame: the head keypoints, the nape->snout axis extended forward to
    the platform-edge plane (vertical line), and a marker where it crosses -
    that crossing point is the single number fig16 tracks over time.
    """
    w = W[ex]
    d = w["raw"]
    sgn = w["sgn"]
    xe = GEOM["right_edge_x"] if sgn > 0 else GEOM["left_edge_x"]   # edge to clear
    times = [-0.6, -0.4, -0.2, 0.0, 0.15]

    t = d.t.to_numpy()
    nx, nz = d.nape_x.to_numpy(), d.nape_z.to_numpy()
    ang = d.ang_nape_snout.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        s = (xe - nx) / np.cos(np.deg2rad(ang))
        hit_zup = nz + s * np.sin(np.deg2rad(ang))
        hit_zup = np.where((s > 0) & (np.abs(np.cos(np.deg2rad(ang))) > MIN_COS)
                           & (np.abs(s) < MAX_REACH_PX), hit_zup, np.nan)

    # crop box: must contain BOTH the head and the fixation point at every shown
    # frame (during the approach the axis points well below head height)
    idx = [int(np.nanargmin(np.abs(t - tt))) for tt in times]
    px = [nx[i] for i in idx if np.isfinite(nx[i])] + [xe]
    py = ([L.IMG - 1 - nz[i] for i in idx if np.isfinite(nz[i])]
          + [L.IMG - 1 - hit_zup[i] for i in idx if np.isfinite(hit_zup[i])])
    x0, x1 = min(px) - 80, max(px) + 80
    y0, y1 = min(py) - 60, max(py) + 60      # image y: y0 = top

    fig, axes = plt.subplots(1, len(times) + 1, figsize=(3.0 * (len(times) + 1), 3.9))
    for ci, tt in enumerate(times):
        ax = axes[ci]
        i = idx[ci]
        fr = int(d.abs_frame.to_numpy()[i])
        ax.imshow(X.grab_frame(2, fr / L.FPS), cmap="gray", vmin=0, vmax=255)
        ax.axvline(xe, color="orange", lw=1.5, ls="-")
        ax.plot(xe, L.IMG - 1 - GEOM["plat_surface_y_up"], ">", color="orange", ms=7)
        for bp, col, ms in [("snout", "red", 5), ("eye", "yellow", 3.5),
                            ("ear", "lime", 3.5), ("head_top", "cyan", 3.5),
                            ("nape", "magenta", 5)]:
            bx, bz = d[f"{bp}_x"].to_numpy()[i], d[f"{bp}_z"].to_numpy()[i]
            if np.isfinite(bx):
                ax.plot(bx, L.IMG - 1 - bz, "o", ms=ms, color=col, mec="k", mew=.4)
        if np.isfinite(hit_zup[i]) and np.isfinite(nx[i]):
            hy = L.IMG - 1 - hit_zup[i]
            ax.plot([nx[i], xe], [L.IMG - 1 - nz[i], hy], "-", color="white", lw=1.4)
            ax.plot([nx[i], xe], [L.IMG - 1 - nz[i], hy], "--", color="tab:blue", lw=1.2)
            ax.plot(xe, hy, "+", ms=16, mew=2.2, color="cyan")
            ax.plot(xe, hy, "o", ms=14, mfc="none", mec="cyan", mew=1.4)
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(f"t = {tt:+.2f} s", fontsize=8)
    axes[0].set_ylabel(f"{ex}  ({'right' if sgn > 0 else 'left'}ward up->up)", fontsize=8)
    axes[0].text(0.02, 0.02,
                 "magenta=nape  red=snout\nwhite/blue = nape->snout axis, extended\n"
                 "orange line = platform-edge plane\n"
                 "orange > = platform top surface\n"
                 "cyan + = fixation point (axis x plane)",
                 transform=axes[0].transAxes, fontsize=6.5, va="bottom", color="w",
                 bbox=dict(fc="k", alpha=0.5, pad=1.5))

    ax = axes[-1]
    ok = np.isfinite(hit_zup)
    ax.plot(t[ok], hit_zup[ok], "-", color="tab:blue", lw=1.4)
    mA = (t >= APPROACH[0]) & (t < APPROACH[1]) & ok
    base = np.nanmedian(hit_zup[mA])
    lo, hi = np.nanpercentile(hit_zup[mA], [5, 95])
    ax.axhspan(lo, hi, color="tab:blue", alpha=0.15,
               label=f"approach 5-95 ({hi - lo:.0f} px)")
    ax.axhline(base, color="tab:blue", lw=0.8, ls=":")
    ax.axhline(GEOM["plat_surface_y_up"], color="orange", lw=1, ls="-")
    ax.text(-0.98, GEOM["plat_surface_y_up"] + 6, "platform top", fontsize=6.5, color="orange")
    hsamp = [hit_zup[i] for i in idx if np.isfinite(hit_zup[i])]
    for tt, i in zip(times, idx):
        if np.isfinite(hit_zup[i]):
            ax.plot(tt, hit_zup[i], "o", ms=6, color="k", zorder=5)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_xlim(-1.0, 0.5)
    ax.set_ylim(min(hsamp) - 90, max(hsamp) + 130)
    ax.set_xlabel("time from onset (s)")
    ax.set_ylabel("fixation height on the\nedge plane (px, up = higher)")
    ax.set_title("the number fig16 tracks\n(here, NOT centred)", fontsize=8)
    ax.legend(fontsize=6.5, loc="lower left")
    ax.grid(True, alpha=0.3)

    fig.suptitle("fig16b  Worked example of the fig16 fixation-point projection, on real "
                 f"Cam2 frames ({ex}).\nThe blue/white line is the nape->snout head axis "
                 "extended to the orange plane; the cyan marker is where it lands. During "
                 "the approach that marker barely moves (shaded band); at takeoff it sweeps "
                 "up and off.", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig16b_fixation_worked_example.png"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    jumps = pd.read_csv(os.path.join(L.V3, "v3_jumps.csv"))
    jumps["ballistic"] = jumps["ballistic"].fillna("").astype(str)
    kp = L.build_keypoints_v3()
    body = L.load_body(2)
    grid = np.arange(-A.PRE_S, A.POST_S + 1e-9, 0.01)

    W = {}
    for r in jumps.itertuples():
        d = A.window(kp, body, r, cam=2)
        if d is None:
            continue
        sgn = A.travel_sign(d)
        dm = A.mirror(d, sgn)
        xt, zt, t_land = A.landing_target(dm, r)
        gap = A.crosses_gap(d)
        W[r.jump] = dict(raw=d, m=dm, sgn=sgn, target=(xt, zt), t_land=t_land,
                         row=r, gap=gap,
                         edge=A.edge_target(sgn) if gap else (np.nan, np.nan))

    df = per_jump_table(W)
    safe_csv(df, os.path.join(L.V3, "v3_stability.csv"))
    show = ["jump", "cls", "ballistic", "sd_approach", "sd_jump", "sd_post",
            "sddet_approach", "pkav_approach", "pkav_jump",
            "ang_range_approach", "bearing_sd_approach", "bearing_range_approach"]
    print("\n--- within-trial head-axis stability (per jump) ---")
    print(df[show].round(1).to_string(index=False))

    print("\nmedian across jumps:")
    for n, _, _ in EP:
        print(f"  {n:>9}: circ SD {df[f'sd_{n}'].median():5.1f} deg   "
              f"detrended {df[f'sddet_{n}'].median():4.1f} deg   "
              f"peak|w| {df[f'pkav_{n}'].median():6.0f} deg/s")
    up = df[df.cls == "up->up"]
    print(f"\nup->up approach: head-axis angle 5-95 range = "
          f"{up['ang_range_approach'].median():.1f} deg (median), "
          f"axis->edge bearing SD = {up['bearing_sd_approach'].median():.1f} deg")
    n_lo = int((df[["sd_approach", "sd_jump"]].max(axis=1) < df.sd_post).sum())
    print(f"{n_lo}/{df.sd_post.notna().sum()} jumps: both approach and jump-epoch SD "
          f"below the post-landing SD")

    ex = _pick_example(W)
    fig_stability(df, W)
    fig_fixation_worked_example(W, ex)
    print(f"worked-example jump: {ex}")
    pf = fig_fixation(W, grid)
    if pf is not None and len(pf):
        safe_csv(pf, os.path.join(L.V3, "v3_fixation_point.csv"))
        print("\n--- fixation-point spread (5-95 px range) ---")
        print(pf.round(1).to_string(index=False))
    print(f"\nfigures -> {FIG}")


if __name__ == "__main__":
    main()
