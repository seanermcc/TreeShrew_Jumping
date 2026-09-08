"""
Cam3-only reproductions of fig06, fig09(a), fig13, fig15, fig16.

Cam3 is the oblique 3/4 view. It is NOT calibrated and its image axes are a
blend of along-tunnel, across-tunnel and height, so every angle here is
approximate - "do your best with the tilt". Two deliberate departures from the
Cam2 pipeline:

  * primary head axis = EYE -> SNOUT, not nape -> snout. Cam3 `nape` is the
    weakest keypoint and is missing through much of the approach on ~half the
    jumps; eye+snout coverage is far better. (fig06 still shows both.)
  * fig09b (aim at the platform front edge) and fig09c (aim at the forepaw
    touchdown) are NOT reproduced: in this oblique view the platform edges are
    foreshortened parallelograms often partly out of frame, and the forepaw at
    contact is occluded/blurred - any annotation would be a +-50 px guess.
    fig09b is replaced by fig09b_cam3 = aim at the LAUNCH POINT (the head's own
    position at onset), which needs no annotation. See the figure caption.

Outputs -> v3/figures/*_cam3.png , v3/v3_cam3_stability.csv
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
import v3_stability as S

FIG = A.FIG
CAM = 3
PRIMARY = "ang_eye_snout"
APP = S.APPROACH
EP = S.EP
CC = A.CLASS_COLORS
GRID = np.arange(-A.PRE_S, A.POST_S + 1e-9, 0.01)
APPROACH_COV_MIN = 0.55        # eye+snout fraction over the approach to use a jump


# --------------------------------------------------------------------------- #
def approach_cov(d):
    t = d.t.to_numpy()
    m = (t >= APP[0]) & (t < APP[1])
    if not m.any():
        return 0.0
    return float((np.isfinite(d.eye_x) & np.isfinite(d.snout_x)).to_numpy()[m].mean())


def build():
    jumps = pd.read_csv(os.path.join(L.V3, "v3_jumps.csv"))
    jumps["ballistic"] = jumps["ballistic"].fillna("").astype(str)
    summ = pd.read_csv(os.path.join(L.V3, "v3_summary.csv")).set_index("jump")
    kp = L.build_keypoints_v3()
    body = L.load_body(2)
    W = {}
    for r in jumps.itertuples():
        d = A.window(kp, body, r, cam=CAM)
        if d is None:
            continue
        sgn = A.travel_sign(d)
        dm = A.mirror(d, sgn)
        xt, zt, t_land = A.landing_target(dm, r)
        # launch point = mirrored head position at onset (t≈0)
        i0 = int(np.nanargmin(np.abs(dm.t.to_numpy())))
        lx, lz = float(dm.head_x.to_numpy()[i0]), float(dm.head_z.to_numpy()[i0])
        W[r.jump] = dict(raw=d, m=dm, sgn=sgn, row=r,
                         cls=r.cls, gap=bool(summ.loc[r.jump, "crosses_gap"]),
                         target=(xt, zt), launch=(lx, lz), t_land=t_land,
                         cov=approach_cov(d))
    return W


def stack(W, col, ids, key="m"):
    return np.vstack([A.resample(W[j][key], col, GRID) for j in ids])


def usable(W, classes=None):
    ids = [j for j, w in W.items() if w["cov"] >= APPROACH_COV_MIN
           and (classes is None or w["cls"] in classes)]
    return ids


# --------------------------------------------------------------------------- #
def fig06_cam3(W):
    by = {c: [j for j in usable(W) if W[j]["cls"] == c] for c in ("up->up", "up->down")}
    by = {c: v for c, v in by.items() if v}
    rows = [("ang_eye_snout", "eye -> snout  (primary for Cam3)"),
            ("ang_nape_snout", "nape -> snout  (sparse in Cam3)")]
    fig, axes = plt.subplots(2, len(by) + 1, figsize=(4.7 * (len(by) + 1), 8.4),
                             sharex=True, squeeze=False)
    for ri, (col, nice) in enumerate(rows):
        for ci, c in enumerate(by):
            ax = axes[ri][ci]
            mat = stack(W, col, by[c])
            for r in mat:
                ax.plot(GRID, r, lw=0.8, alpha=0.4, color=CC[c])
            mu = L.circmean_deg(mat, axis=0)
            se = L.circsem_deg(mat, axis=0)
            nn = np.sum(np.isfinite(mat), axis=0)
            mu = np.where(nn >= 3, mu, np.nan)
            ax.plot(GRID, mu, lw=2.6, color="black", label=f"circ mean (n={len(mat)})")
            ax.fill_between(GRID, mu - se, mu + se, color="black", alpha=0.2, lw=0)
            ax.axvline(0, color="k", ls="--", lw=1)
            ax.axhline(0, color="tab:gray", lw=0.8)
            ax.set_xlim(-1.0, 1.5)
            ax.set_ylim(-180, 120)
            ax.set_title(f"{c}  (n={len(by[c])})\n{nice}", fontsize=9)
            ax.legend(fontsize=7, loc="lower left")
            if ci == 0:
                ax.set_ylabel("Cam3 image angle from +x (deg)\n"
                              "mirrored so travel = +x")
        ax = axes[ri][-1]
        for c in by:
            mat = stack(W, col, by[c])
            nn = np.sum(np.isfinite(mat), axis=0)
            ax.plot(GRID, np.where(nn >= 3, L.circmean_deg(mat, axis=0), np.nan),
                    lw=2.2, color=CC[c], label=f"{c} (n={len(by[c])})")
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.axhline(0, color="tab:gray", lw=0.8)
        ax.set_xlim(-1.0, 1.5)
        ax.set_ylim(-180, 120)
        ax.set_title("class means overlaid", fontsize=9)
        ax.legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("time from jump onset (s)")
    fig.suptitle("fig06_cam3  Head orientation from Cam3 (oblique view, uncalibrated).\n"
                 "Compare the SHAPE to fig06, not the absolute angles - Cam3's zero and "
                 "scale differ. Mirrored so every jump travels +x.", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig06_cam3.png"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
def target_error3(w, which):
    d = w["m"]
    tx, tz = w[which]
    if not np.isfinite(tx):
        return np.full(len(GRID), np.nan)
    dx = (tx - d.head_x).to_numpy()
    dz = (tz - d.head_z).to_numpy()
    dist = np.hypot(dx, dz)
    err = L.wrap180(pd.to_numeric(d[PRIMARY], errors="coerce").to_numpy() - L.ang_deg(dx, dz))
    keep = (dist >= A.TARGET_MIN_DIST) & (d.t.to_numpy() <= w["t_land"] + 0.15)
    return A.resample(d[["t"]].assign(e=np.where(keep, err, np.nan)), "e", GRID)


def fig09_cam3(W, which, fname, subtitle):
    by = {c: [j for j in usable(W, (c,)) if np.isfinite(W[j][which][0])]
          for c in ("up->up", "up->down")}
    by = {c: v for c, v in by.items() if v}
    disp = (GRID >= -1.0) & (GRID <= 1.2)
    fig, axes = plt.subplots(1, len(by) + 1, figsize=(5.2 * (len(by) + 1), 4.6))
    per = {}
    for ci, c in enumerate(by):
        ax = axes[ci]
        mat = np.vstack([target_error3(W[j], which) for j in by[c]])
        per[c] = mat
        for r in mat:
            ax.plot(GRID[disp], r[disp], lw=0.9, alpha=0.5, color=CC[c])
        nn = np.sum(np.isfinite(mat), axis=0)
        mu = np.where(nn >= 3, L.circmean_deg(mat, axis=0), np.nan)
        ax.plot(GRID[disp], mu[disp], lw=2.6, color="black", label=f"circ mean (n={len(mat)})")
        ax.axhline(0, color="tab:green", lw=1.4)
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.set_ylim(-150, 170)
        ax.set_xlim(-1.0, 1.2)
        ax.set_title(f"{c}  (n={len(by[c])})", fontsize=9)
        ax.set_ylabel("Cam3 eye->snout error vs target (deg)\n0 = pointing at it")
        ax.set_xlabel("time from onset (s)")
        ax.legend(fontsize=7)
        # approach mean |err|
        a = np.abs(mat[:, (GRID >= -0.6) & (GRID < -0.1)])
        v = np.nanmean(a, axis=1)
        v = v[np.isfinite(v)]
        ax.text(0.02, 0.03, f"approach |err| = {np.mean(v):.0f} +- {np.std(v):.0f} deg",
                transform=ax.transAxes, fontsize=8,
                bbox=dict(fc="w", alpha=0.7))
    ax = axes[-1]
    for c in by:
        mat = per[c]
        nn = np.sum(np.isfinite(mat), axis=0)
        ax.plot(GRID[disp], np.where(nn >= 3, np.nanmean(np.abs(mat), axis=0), np.nan)[disp],
                lw=2.2, color=CC[c], label=f"{c} (n={len(by[c])})")
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_xlim(-1.0, 1.2)
    ax.set_title("mean |aiming error| (lower = better aimed)", fontsize=9)
    ax.set_ylabel("deg")
    ax.set_xlabel("time from onset (s)")
    ax.legend(fontsize=7)
    fig.suptitle(f"{fname[:-4]}  Head aiming from Cam3 (eye->snout axis, oblique view).\n"
                 + subtitle, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)


# --------------------------------------------------------------------------- #
def fig13_cam3(W):
    reps = []
    for c in ("up->up", "up->down", "down->up"):
        ids = [j for j in W if W[j]["cls"] == c]
        if ids:
            reps.append(max(ids, key=lambda j: W[j]["cov"]))
    times = [-0.6, -0.3, -0.1, 0.0, 0.15, 0.4]
    fig, axes = plt.subplots(len(reps), len(times) + 1,
                             figsize=(2.15 * (len(times) + 1), 2.4 * len(reps)),
                             squeeze=False)
    kp = L.build_keypoints_v3()
    for ri, j in enumerate(reps):
        d = W[j]["raw"]
        k = kp[(kp["jump"] == j) & (kp.cam == CAM)].set_index("abs_frame")
        for ci, tt in enumerate(times):
            ax = axes[ri][ci]
            i = int(np.nanargmin(np.abs(d.t.to_numpy() - tt)))
            fr = int(d.abs_frame.to_numpy()[i])
            ax.imshow(X.grab_frame(CAM, fr / L.FPS), cmap="gray", vmin=0, vmax=255)
            if fr in k.index:
                row = k.loc[fr]
                for bp, col in [("snout", "red"), ("eye", "yellow"), ("ear", "lime"),
                                ("head_top", "cyan"), ("nape", "magenta")]:
                    if np.isfinite(row[f"{bp}_x"]):
                        ax.plot(row[f"{bp}_x"], row[f"{bp}_y"], "o", ms=3.5,
                                color=col, mec="k", mew=0.4)
                if np.isfinite(row["eye_x"]) and np.isfinite(row["snout_x"]):
                    dx, dy = row["snout_x"] - row["eye_x"], row["snout_y"] - row["eye_y"]
                    ax.plot([row["eye_x"], row["eye_x"] + 4 * dx],
                            [row["eye_y"], row["eye_y"] + 4 * dy], "-", color="white", lw=1)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            ax.set_title(f"t={tt:+.2f}s", fontsize=7)
            if ci == 0:
                ax.set_ylabel(f"{j}\n{W[j]['cls']}", fontsize=7)
        ax = axes[ri][-1]
        ax.plot(d.t, d.head_speed, lw=1.3, color=CC.get(W[j]["cls"], "gray"))
        for tt in times:
            ax.axvline(tt, color="gray", lw=0.6, ls=":")
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.set_title("Cam3 head speed (px/s)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)
    fig.suptitle("fig13_cam3  Representative jumps, Cam3 frames with DLC keypoints + the "
                 "eye->snout axis (white).\nred=snout yellow=eye lime=ear cyan=head_top "
                 "magenta=nape", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig13_cam3.png"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
def _seg(d, a, b, col):
    m = (d.t >= a) & (d.t < b)
    return d.t.to_numpy()[m], pd.to_numeric(d[col], errors="coerce").to_numpy()[m]


def fig15_cam3(W):
    ids = usable(W)
    rows = []
    for j in ids:
        d = W[j]["m"]
        rec = dict(jump=j, cls=W[j]["cls"], cov=round(W[j]["cov"], 2))
        for n, a, b in EP:
            t, ang = _seg(d, a, b, PRIMARY)
            rec[f"sd_{n}"] = S.circ_sd_deg(ang)
            rec[f"sddet_{n}"] = S.detrended_sd_deg(t, ang)
            rec[f"pkav_{n}"] = S.peak_ang_vel(t, ang)
        ta, aa = _seg(d, *APP, PRIMARY)
        rec["ang_range_approach"] = S.pctl_range(aa)
        rows.append(rec)
    df = pd.DataFrame(rows)
    S.safe_csv(df, os.path.join(L.V3, "v3_cam3_stability.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    ecol = [S.EP_COL[n] for n, _, _ in EP]
    S._strip_box(axes[0], [df[f"sd_{n}"].to_numpy() for n, _, _ in EP],
                 [n for n, _, _ in EP], ecol,
                 "circ SD of Cam3 eye->snout angle (deg)",
                 "Within-trial head-axis jitter, Cam3")
    S._strip_box(axes[1], [df[f"sddet_{n}"].to_numpy() for n, _, _ in EP],
                 [n for n, _, _ in EP], ecol,
                 "detrended SD (deg)", "drift removed")
    S._strip_box(axes[2], [df[f"pkav_{n}"].to_numpy() for n, _, _ in EP],
                 [n for n, _, _ in EP], ecol,
                 "peak |angular velocity| (deg/s, 95th pct)",
                 "how fast the Cam3 head axis rotates", logy=True)
    up = df[df.cls == "up->up"]
    txt = (f"Cam3, medians across {len(df)} jumps ({len(up)} up->up):   "
           f"approach circ SD {df.sd_approach.median():.1f} deg "
           f"(up->up {up.sd_approach.median():.1f})   |   "
           f"jump {df.sd_jump.median():.1f}   |   post {df.sd_post.median():.1f}   |   "
           f"up->up approach 5-95 angle range {up.ang_range_approach.median():.1f} deg")
    fig.text(0.5, 0.02, txt, ha="center", fontsize=8, family="monospace")
    fig.suptitle("fig15_cam3  Cam3 within-trial head-axis stability (eye->snout). "
                 "Same metric as fig15, different camera.\n"
                 "A low approach SD here too = the stillness is not a Cam2 projection "
                 "artefact (see fig17).", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(FIG, "fig15_cam3.png"))
    plt.close(fig)
    return df


# --------------------------------------------------------------------------- #
def project3(d, xe):
    if not np.isfinite(xe):
        return np.full(len(d), np.nan)
    hx, hz = d.head_x.to_numpy(), d.head_z.to_numpy()
    th = np.deg2rad(pd.to_numeric(d[PRIMARY], errors="coerce").to_numpy())
    cx, sz = np.cos(th), np.sin(th)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = (xe - hx) / cx
        hit = hz + s * sz
        bad = ~np.isfinite(s) | (s <= 0) | (np.abs(cx) < S.MIN_COS) | (np.abs(s) > S.MAX_REACH_PX)
    return np.where(bad, np.nan, hit)


def fig16_cam3(W):
    ids = [j for j in usable(W) if np.isfinite(W[j]["target"][0])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw=dict(width_ratios=[1.5, 1]))
    ax = axes[0]
    per = []
    for j in ids:
        d = W[j]["m"]
        xe = W[j]["target"][0]
        hit = project3(d, xe)
        t = d.t.to_numpy()
        mA = (t >= APP[0]) & (t < APP[1]) & np.isfinite(hit)
        if mA.sum() < 5:
            continue
        base = np.nanmedian(hit[mA])
        ax.plot(t, hit - base, lw=1, color=CC.get(W[j]["cls"], "gray"), alpha=0.6)

        def rng(a, b):
            m = (t >= a) & (t < b) & np.isfinite(hit)
            return S.pctl_range(hit[m]) if m.sum() >= 4 else np.nan
        per.append(dict(jump=j, cls=W[j]["cls"], approach=rng(*APP),
                        jump_ep=rng(0, 0.2), post=rng(0.2, 1.0)))
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlim(-1.0, 0.6)
    ax.set_ylim(-160, 160)
    ax.set_xlabel("time from onset (s)")
    ax.set_ylabel("Cam3 fixation point on the plane\n(px, centred on approach median)")
    ax.set_title("Cam3 eye->snout axis projected to a vertical plane at the\n"
                 "measured landing head-x; flat in the approach = held gaze", fontsize=9)
    pf = pd.DataFrame(per)
    if len(pf):
        S._strip_box(axes[1], [pf.approach.to_numpy(), pf.jump_ep.to_numpy(), pf.post.to_numpy()],
                     ["approach", "jump", "post"], ["tab:blue", "tab:orange", "tab:green"],
                     "5-95 range of the fixation point (px)",
                     "spread by epoch", logy=True)
        S.safe_csv(pf, os.path.join(L.V3, "v3_cam3_fixation.csv"))
        fig.text(0.5, 0.01, f"Cam3 fixation 5-95 range (median px): "
                 f"approach {pf.approach.median():.0f}  |  jump {pf.jump_ep.median():.0f}  |  "
                 f"post {pf.post.median():.0f}   (n={len(pf)})",
                 ha="center", fontsize=8.5, family="monospace")
    fig.suptitle("fig16_cam3  Cam3 fixation point / retinal slip. Rougher than fig16 "
                 "(oblique, uncalibrated) but same test.", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(FIG, "fig16_cam3.png"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    W = build()
    cov = pd.DataFrame([(j, w["cls"], round(w["cov"], 2)) for j, w in W.items()],
                       columns=["jump", "cls", "approach_cov"])
    print("Cam3 approach eye+snout coverage:")
    print(cov.to_string(index=False))
    print(f"\nusing {len(usable(W))} jumps (cov >= {APPROACH_COV_MIN})")

    fig06_cam3(W)
    fig09_cam3(W, "target", "fig09_cam3.png",
               "Target = the MEASURED position where the head settles after landing "
               "(no annotation needed).")
    fig09_cam3(W, "launch", "fig09b_cam3.png",
               "Target = the LAUNCH POINT (head position at onset). Cam3 replacement for "
               "fig09b: the platform edge cannot be annotated reliably in this oblique "
               "view, so we test aiming at the spot the animal launches from instead.")
    fig13_cam3(W)
    df = fig15_cam3(W)
    fig16_cam3(W)

    print("\n--- fig15_cam3 per-jump ---")
    print(df[["jump", "cls", "cov", "sd_approach", "sd_jump", "sd_post",
              "ang_range_approach"]].round(1).to_string(index=False))
    print(f"\nfigures -> {FIG}")


if __name__ == "__main__":
    main()
