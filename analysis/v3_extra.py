"""
Extra v3 analyses beyond the core request.

  fig10  free-flight parabola fit -> an ACTUAL PIXEL SCALE.
         While the animal is airborne the only vertical force is gravity, so
         head height must follow z(t) = z0 + v0 t - 0.5 g t^2 with the same g
         in every jump. Fitting g in px/s^2 and dividing by 9806.65 mm/s^2
         gives px/mm -- an independent, physics-based calibration for a rig
         that has no calibration target. Agreement of g ACROSS jumps is the
         evidence that it is real; scatter would mean the fits are picking up
         contact phases instead of free flight.

  fig11  head angular velocity (how fast the head re-aims), by class.

  fig12  user's ballistic label (y/n) tested against the kinematics.

  fig13  representative frames with tracking, for eyeballing the numbers.

Run after v3_analysis.py.
"""
import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import v3_lib as L
import v3_analysis as A

G_MM_S2 = 9806.65
FIG = A.FIG


# --------------------------------------------------------------------------- #
MIN_SAG_PX = 8.0        # a parabola must actually curve, not just look linear


def flight_window(d, t_land):
    """Longest window after onset consistent with free flight.

    Requires: downward curvature, R^2 >= 0.985, and a parabolic sag of at least
    MIN_SAG_PX. The sag test matters -- a slow straight-line descent (walking
    down) also fits a parabola almost perfectly, but with a ~ 0, and would
    otherwise contaminate the gravity estimate with meaningless small values.
    """
    t = d.t.to_numpy()
    z = d.head_z.to_numpy()
    m = (t >= -0.25) & (t <= min(t_land + 0.10, t[-1]))
    t, z = t[m], z[m]
    ok = np.isfinite(z)
    t, z = t[ok], z[ok]
    n = len(t)
    if n < 14:
        return None
    best = None
    for win in range(n, 13, -2):                   # longest first
        for i in range(0, n - win + 1, 2):
            tt, zz = t[i:i + win], z[i:i + win]
            span = tt[-1] - tt[0]
            if span < 0.12:
                continue
            c = np.polyfit(tt, zz, 2)
            if c[0] >= 0:
                continue
            g = -2 * c[0]
            if g * span ** 2 / 8.0 < MIN_SAG_PX:   # too straight to be a fit of g
                continue
            pred = np.polyval(c, tt)
            tot = np.sum((zz - zz.mean()) ** 2)
            r2 = 1 - np.sum((zz - pred) ** 2) / tot if tot > 0 else 0
            if r2 < 0.985:
                continue
            if best is None or span > best[0]:
                best = (span, tt[0], tt[-1], g, r2, len(tt))
        if best is not None:                       # longest window wins
            break
    return best


def gap_dwell(w):
    """Time the head spends over the gap, and how far it falls while there.

    A body in free flight cannot hover: over t seconds it must drop
    0.5*g*t^2. Long dwell with a small drop is proof of SUPPORT.
    """
    d = w["raw"]
    t, x, z = d.t.to_numpy(), d.head_x.to_numpy(), d.head_z.to_numpy()
    ing = (np.isfinite(x) & (x > A.GEOM["left_edge_x"]) & (x < A.GEOM["right_edge_x"])
           & (t > -0.5) & (t < 1.0))
    if ing.sum() < 3:
        return None
    tg, zg = t[ing], z[ing]
    return dict(dur_s=float(tg[-1] - tg[0]),
                drop_px=float(np.nanmax(zg) - np.nanmin(zg)))


def fastest_window(d, half=0.10):
    """Fit z(t) over the +-`half` s around onset -- the most jump-like moment."""
    t, z = d.t.to_numpy(), d.head_z.to_numpy()
    m = (t >= -half) & (t <= half) & np.isfinite(z)
    if m.sum() < 8:
        return None
    c = np.polyfit(t[m], z[m], 2)
    pred = np.polyval(c, t[m])
    tot = np.sum((z[m] - z[m].mean()) ** 2)
    r2 = 1 - np.sum((z[m] - pred) ** 2) / tot if tot > 0 else np.nan
    return dict(az=float(2 * c[0]), r2=float(r2), n=int(m.sum()))


def fig_ballistic_physics(W):
    """Test each jump against the physics of free flight.

    This started as an attempt to CALIBRATE the rig by fitting gravity to the
    airborne head arc (g in px/s^2 / 9806.65 mm/s^2 -> px/mm). That failed: see
    the printout. The reason it failed is itself the interesting result, so the
    figure now reports the ballistic test rather than a scale.
    """
    rows = []
    for j, w in W.items():
        got = flight_window(w["raw"], w["t_land"])
        gd = gap_dwell(w)
        fw = fastest_window(w["raw"])
        rows.append(dict(
            jump=j, cls=w["row"].cls, ballistic=str(w["row"].ballistic or "?"),
            g_px_s2=round(got[3]) if got else np.nan,
            fit_span_ms=round((got[2] - got[1]) * 1000) if got else np.nan,
            fit_r2=round(got[4], 4) if got else np.nan,
            az_onset=round(fw["az"]) if fw else np.nan,
            az_r2=round(fw["r2"], 3) if fw else np.nan,
            gap_dwell_s=round(gd["dur_s"], 2) if gd else np.nan,
            gap_drop_px=round(gd["drop_px"]) if gd else np.nan))
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    # A: dwell vs drop, with free-fall curves for candidate pixel scales
    ax = axes[0]
    g = df.dropna(subset=["gap_dwell_s"])
    for r in g.itertuples():
        ax.scatter(r.gap_dwell_s, r.gap_drop_px, s=55, color=A.CLASS_COLORS[r.cls],
                   marker="o" if r.ballistic == "y" else "s",
                   edgecolor="k", linewidth=.6, zorder=3)
        ax.annotate(r.jump[3:], (r.gap_dwell_s, r.gap_drop_px), fontsize=6,
                    xytext=(4, 3), textcoords="offset points")
    ts = np.linspace(0.05, 1.6, 100)
    for scale, style in [(1.0, "-"), (0.5, "--"), (0.25, ":")]:
        ax.plot(ts, 0.5 * G_MM_S2 * scale * ts ** 2, style, color="black", lw=1,
                label=f"free fall @ {scale} px/mm")
    ax.set_ylim(0, 420)
    ax.set_xlabel("time head spends over the gap (s)")
    ax.set_ylabel("vertical drop while over the gap (px)")
    ax.set_title("Hovering over a gap is impossible in free flight")
    ax.legend(fontsize=7, loc="upper left")

    # B: vertical acceleration in the +-100 ms around onset
    ax = axes[1]
    d2 = df.dropna(subset=["az_onset"]).sort_values("az_onset")
    ax.barh(range(len(d2)), d2.az_onset,
            color=[A.CLASS_COLORS[c] for c in d2.cls], alpha=.85,
            edgecolor="k", linewidth=.5)
    ax.set_yticks(range(len(d2)))
    ax.set_yticklabels([f"{r.jump} {r.cls[:7]} bal={r.ballistic}" for r in d2.itertuples()],
                       fontsize=6.5)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("vertical acceleration, ±100 ms around onset (px/s²)")
    ax.set_title("Vertical acceleration at onset\n"
                 "+ve = arresting a descent (landing), -ve = accelerating downward",
                 fontsize=9)

    # C: the failed calibration, shown as the evidence it failed
    ax = axes[2]
    d3 = df.dropna(subset=["g_px_s2"]).sort_values("g_px_s2")
    if len(d3):
        ax.barh(range(len(d3)), d3.g_px_s2,
                color=[A.CLASS_COLORS[c] for c in d3.cls], alpha=.85,
                edgecolor="k", linewidth=.5)
        ax.set_yticks(range(len(d3)))
        ax.set_yticklabels([f"{r.jump} bal={r.ballistic} R²={r.fit_r2:.3f}"
                            for r in d3.itertuples()], fontsize=6.5)
        lo, hi = d3.g_px_s2.min(), d3.g_px_s2.max()
        ax.set_title(f"Best parabola fits disagree {hi/lo:.1f}x\n"
                     "-> not gravity -> NO metric scale recoverable")
    ax.set_xlabel("fitted 'g' (px/s²)")

    fig.suptitle("Is the animal actually ballistic? Circles = user labelled ballistic 'y', "
                 "squares = 'n'.\nIf these were true leaps, panel C would give one shared "
                 "value of g and hence a pixel scale. It does not.", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig10_ballistic_physics.png"))
    plt.close(fig)
    df.to_csv(os.path.join(L.V3, "v3_ballistic_physics.csv"), index=False)
    return df

    # per-jump fit quality plot
    n = len(df)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.6 * nrow), squeeze=False)
    for ax, r in zip(axes.ravel(), df.itertuples()):
        d = W[r.jump]["raw"]
        m = (d.t >= r.t0 - 0.35) & (d.t <= r.t1 + 0.35)
        ax.plot(d.t[m], d.head_z[m], ".", ms=2.5, color="tab:gray")
        mf = (d.t >= r.t0) & (d.t <= r.t1)
        tt, zz = d.t[mf].to_numpy(), d.head_z[mf].to_numpy()
        ok = np.isfinite(zz)
        if ok.sum() > 3:
            c = np.polyfit(tt[ok], zz[ok], 2)
            ts = np.linspace(tt[ok][0], tt[ok][-1], 80)
            ax.plot(tt[ok], zz[ok], ".", ms=3.5, color=A.CLASS_COLORS[r.cls])
            ax.plot(ts, np.polyval(c, ts), "-", lw=1.6, color="k")
        ax.set_title(f"{r.jump} {r.cls}\ng={r.g_px_s2:.0f} px/s²  R²={r.r2:.3f}", fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Per-jump free-flight fits: head height (px) vs time (s). "
                 "Black = fitted parabola, coloured = frames used.", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig10b_gravity_fits_per_jump.png"))
    plt.close(fig)

    df.to_csv(os.path.join(L.V3, "v3_gravity_fits.csv"), index=False)
    return df


# --------------------------------------------------------------------------- #
def fig_angular_velocity(grid, stack, by_cls):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    for ax, (col, nice) in zip(axes, [("ang_nape_snout", "head axis (nape->snout)"),
                                      ("ang_eye_snout", "muzzle (eye->snout)")]):
        for c, ids in by_cls.items():
            mat, _ = stack(col, key="m", subset=set(ids))
            un = A.unwrap_deg(mat)
            av = np.gradient(un, axis=1) * L.FPS      # deg/s
            av = np.abs(av)
            m, e = np.nanmean(av, axis=0), A.sem(av, axis=0)
            ax.plot(grid, m, lw=2, color=A.CLASS_COLORS[c], label=f"{c} (n={len(ids)})")
            ax.fill_between(grid, m - e, m + e, color=A.CLASS_COLORS[c],
                            alpha=0.2, linewidth=0)
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.set_title(f"|angular velocity|: {nice}")
        ax.set_xlabel("time from jump onset (s)")
    axes[0].set_ylabel("deg/s")
    axes[0].legend(fontsize=7)
    fig.suptitle("How fast the head re-aims. Peaks before onset = aiming; "
                 "peaks after = landing/righting.")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig11_angular_velocity.png"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
def fig_ballistic(grid, stack, W):
    """Does the manual ballistic y/n label separate on the kinematics?"""
    yes = [j for j in W if str(W[j]["row"].ballistic).lower() == "y"]
    no = [j for j in W if str(W[j]["row"].ballistic).lower() == "n"]
    if not yes or not no:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    for ids, lab, col in [(yes, "ballistic = y", "tab:red"), (no, "ballistic = n", "tab:blue")]:
        mat, _ = stack("head_speed", subset=set(ids))
        m, e = np.nanmean(mat, axis=0), A.sem(mat, axis=0)
        axes[0].plot(grid, m, lw=2.2, color=col, label=f"{lab} (n={len(ids)})")
        axes[0].fill_between(grid, m - e, m + e, color=col, alpha=0.22, linewidth=0)
    axes[0].axvline(0, color="k", ls="--", lw=1)
    axes[0].set_title("Head speed by the manual ballistic label")
    axes[0].set_ylabel("head speed (px/s)")
    axes[0].set_xlabel("time from onset (s)")
    axes[0].legend(fontsize=8)

    groups = []
    for ids in (yes, no):
        groups.append(np.array([[A.epoch_means(W[j]["raw"], "head_speed")[n]
                                 for n, _, _ in A.EPOCHS] for j in ids]))
    A.bars(axes[1], groups, ["ballistic y", "ballistic n"], ["tab:red", "tab:blue"],
           "Epoch means by ballistic label", "head speed (px/s)")
    axes[1].set_xlabel("epoch")

    pk_y = [np.nanmax(W[j]["raw"].head_speed) for j in yes]
    pk_n = [np.nanmax(W[j]["raw"].head_speed) for j in no]
    axes[2].boxplot([pk_y, pk_n], labels=["y", "n"], patch_artist=True,
                    boxprops=dict(facecolor="tab:red", alpha=.5))
    rng = np.random.default_rng(2)
    for i, v in enumerate([pk_y, pk_n], 1):
        axes[2].plot(np.full(len(v), i) + rng.uniform(-.07, .07, len(v)), v, "o",
                     ms=5, color="black", alpha=0.7)
    axes[2].set_ylabel("peak head speed (px/s)")
    axes[2].set_xlabel("ballistic label")
    axes[2].set_title("Peak speed vs manual label")
    fig.suptitle("Testing the manual 'ballistic' call against the tracking. "
                 "One jump has no label and is excluded.")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig12_ballistic_label.png"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
def fig_representative(W, jumps_to_show, kp):
    """Real frames with the tracking drawn, next to the speed trace, so the
    plotted numbers can be checked against the pixels they came from."""
    times = [-0.5, -0.1, 0.0, 0.1, 0.3, 0.8]
    n = len(jumps_to_show)
    fig, axes = plt.subplots(n, len(times) + 1,
                             figsize=(2.05 * (len(times) + 1), 2.25 * n), squeeze=False)
    for ri, j in enumerate(jumps_to_show):
        w = W[j]
        r = w["row"]
        k = kp[(kp["jump"] == j) & (kp.cam == 2)].set_index("abs_frame")
        for ci, tt in enumerate(times):
            ax = axes[ri][ci]
            fr = int(round(r.onset_frame + tt * L.FPS))
            img = grab_frame(2, fr / L.FPS)
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            if fr in k.index:
                row = k.loc[fr]
                for bp, col in [("snout", "red"), ("eye", "yellow"), ("ear", "lime"),
                                ("head_top", "cyan"), ("nape", "magenta")]:
                    x, y = row[f"{bp}_x"], row[f"{bp}_y"]
                    if np.isfinite(x):
                        ax.plot(x, y, "o", ms=3.5, color=col, mec="k", mew=0.4)
                if np.isfinite(row["head_x"]):
                    ax.plot(row["head_x"], row["head_y"], "o", ms=6, mfc="none",
                            mec="white", mew=1.2)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            ax.set_title(f"t={tt:+.1f}s", fontsize=7)
            if ci == 0:
                ax.set_ylabel(f"{j}\n{r.cls}", fontsize=7)
        ax = axes[ri][-1]
        d = w["raw"]
        ax.plot(d.t, d.head_speed, lw=1.4, color=A.CLASS_COLORS[r.cls])
        for tt in times:
            ax.axvline(tt, color="gray", lw=0.6, ls=":")
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.set_title("head speed (px/s)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Representative jumps: Cam2 frames with the DLC keypoints actually used, "
                 "next to the head-speed trace they produce.\n"
                 "white ring = head centroid; dots = snout/eye/ear/head_top/nape", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig13_representative_frames.png"))
    plt.close(fig)


def fig_paw_validation(kp):
    """Every annotated forepaw touchdown: the actual Cam2 frame with the marked
    point + the DLC head keypoints, so the annotation can be eyeballed."""
    pc = pd.read_csv(os.path.join(L.V3, "paw_contacts.csv"))
    n = len(pc)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 3.7 * nrow), squeeze=False)
    for ax, r in zip(axes.ravel(), pc.itertuples()):
        img = grab_frame(2, r.contact_frame / L.FPS)
        # crop 300x300 around the paw point
        x0, y0 = int(r.paw_x_img - 150), int(r.paw_y_img - 150)
        x0, y0 = max(0, min(x0, L.IMG - 300)), max(0, min(y0, L.IMG - 300))
        ax.imshow(img[y0:y0 + 300, x0:x0 + 300], cmap="gray", vmin=0, vmax=255,
                  extent=[x0, x0 + 300, y0 + 300, y0])
        ax.plot(r.paw_x_img, r.paw_y_img, "+", ms=18, mew=2.5, color="cyan")
        ax.plot(r.paw_x_img, r.paw_y_img, "o", ms=16, mfc="none", mec="cyan", mew=1.5)
        k = kp[(kp["jump"] == r.jump) & (kp.cam == 2)
               & (kp.abs_frame == r.contact_frame)]
        if len(k):
            row = k.iloc[0]
            for bp, col in [("snout", "red"), ("eye", "yellow"), ("nape", "magenta")]:
                if np.isfinite(row[f"{bp}_x"]):
                    ax.plot(row[f"{bp}_x"], row[f"{bp}_y"], "o", ms=5, color=col, mec="k", mew=.4)
            if np.isfinite(row["nape_x"]) and np.isfinite(row["snout_x"]):
                ax.plot([row["nape_x"], row["snout_x"]], [row["nape_y"], row["snout_y"]],
                        "-", color="white", lw=1.2)
        ax.set_title(f"{r.jump}  {r.surface}  conf={r.confidence}\n"
                     f"f{r.contact_frame}  paw=({r.paw_x_img},{r.paw_y_img})", fontsize=7.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Forepaw touchdown annotations (cyan +) vs the actual Cam2 frame.\n"
                 "red=snout  yellow=eye  magenta=nape  white line=head axis.  "
                 "These points define the 'paw' target in fig09c.", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig14_paw_annotation_check.png"))
    plt.close(fig)


def grab_frame(cam, t):
    cmd = [L.FFMPEG, "-v", "error", "-nostdin", "-ss", f"{t:.3f}", "-i", L.SRC.format(cam),
           "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    p = subprocess.run(cmd, capture_output=True)
    buf = p.stdout
    if len(buf) < L.IMG * L.IMG:
        return np.zeros((L.IMG, L.IMG), np.uint8)
    return np.frombuffer(buf[:L.IMG * L.IMG], np.uint8).reshape(L.IMG, L.IMG)


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
        W[r.jump] = dict(raw=d, m=dm, sgn=sgn, target=(xt, zt), t_land=t_land, row=r)

    def stack(col, key="raw", subset=None):
        ids = [j for j in W if subset is None or j in subset]
        return np.vstack([A.resample(W[j][key], col, grid) for j in ids]), ids

    by_cls = {c: [j for j in W if W[j]["row"].cls == c] for c in A.CLASS_COLORS}
    by_cls = {c: v for c, v in by_cls.items() if v}

    g = fig_ballistic_physics(W)
    print("\n--- ballistic / free-flight test ---")
    print(g.to_string(index=False))
    gd = g.dropna(subset=["gap_dwell_s"])
    if len(gd):
        print(f"\nHead dwells over the gap for {gd.gap_dwell_s.min():.2f}-"
              f"{gd.gap_dwell_s.max():.2f} s while dropping only "
              f"{gd.gap_drop_px.min():.0f}-{gd.gap_drop_px.max():.0f} px.")
        worst = gd.loc[gd.gap_dwell_s.idxmax()]
        print(f"  e.g. {worst.jump}: {worst.gap_dwell_s:.2f} s over the gap, "
              f"drop {worst.gap_drop_px:.0f} px. Unsupported, that alone would need "
              f"a scale of {2*worst.gap_drop_px/(G_MM_S2*worst.gap_dwell_s**2):.3f} px/mm "
              f"(i.e. the 242 px gap would be "
              f"{242/(2*worst.gap_drop_px/(G_MM_S2*worst.gap_dwell_s**2)):.0f} mm wide).")
    gg = g.dropna(subset=["g_px_s2"])
    if len(gg) > 1:
        print(f"\nParabola fits span {gg.g_px_s2.min():.0f}-{gg.g_px_s2.max():.0f} px/s² "
              f"({gg.g_px_s2.max()/gg.g_px_s2.min():.1f}x). Gravity is one number, so these "
              f"windows are NOT free flight:\n  no metric calibration is recoverable this "
              f"way -- a calibration target is still required.")

    fig_angular_velocity(grid, stack, by_cls)
    fig_ballistic(grid, stack, W)

    # Representative jump per class. The automatic pick is "best head-tracking
    # coverage", but for up->up that returns v3_01 -- a marginal ~200 px "hop up"
    # (confirmed=maybe, ballistic=n) that reads as a non-jump. Pin that slot to
    # v3_10 instead: a clean, confirmed, ballistic gap crossing with all five
    # keypoints visible in every sampled frame.
    REP_PIN = {"up->up": "v3_10"}
    rep = []
    for c in ("up->up", "up->down", "down->up"):
        ids = [j for j in W if W[j]["row"].cls == c]
        if not ids:
            continue
        pin = REP_PIN.get(c)
        rep.append(pin if pin in ids else
                   max(ids, key=lambda j: W[j]["raw"].head_raw_ok.mean()))
    fig_representative(W, rep, kp)
    print(f"\nrepresentative jumps shown: {rep}")

    fig_paw_validation(kp)
    print(f"figures -> {FIG}")


if __name__ == "__main__":
    main()
