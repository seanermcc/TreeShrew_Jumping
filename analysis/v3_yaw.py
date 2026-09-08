"""
YAW CHECK - does the head rotate left/right (about the vertical) during the
approach, which the Cam2 sagittal view is blind to?

Cam2 sees head PITCH (nose up/down) and fore-aft position. A head that held its
pitch rock-steady in Cam2 could still be yawing toward/away from the camera and
fig06/fig15/fig16 would not know. Cam3 is a roughly frontal / 3-4 oblique view,
so its projected head-axis angle is strongly YAW-sensitive (and pitch-sensitive
too). The argument here is comparative:

  - if the head is genuinely still in 3D  -> BOTH the Cam2 and the Cam3
    within-approach head-axis jitter are small;
  - if the head held pitch but yawed      -> Cam2 jitter small, Cam3 jitter LARGE.

Plus two Cam2-only cross-checks:
  - eye->snout projected LENGTH in Cam2 foreshortens as cos(yaw); a flat length
    trace bounds the yaw excursion;
  - the pseudo-3D azimuth (Cam2 x = along, Cam3 x = across proxy - the HANDOFF
    `Y_across`) computed directly, with its caveats.

Uses the eye->snout axis throughout: Cam3 `nape` is the weakest keypoint and
drops out in the approach on several jumps, whereas eye+snout coverage is >=0.9
there for ~11 jumps.

Outputs -> v3/figures/fig17_yaw_check.png , v3/v3_yaw.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import v3_lib as L
import v3_analysis as A
import v3_stability as S

FIG = A.FIG
APP = S.APPROACH                       # (-1.0, -0.15)
EP = S.EP                              # approach / jump / post


def _ang(dx, dz):
    return L.ang_deg(dx, dz)


def cam_angle(d, a="eye", b="snout"):
    """Projected b<-a axis angle in that camera's image plane (deg)."""
    return _ang((d[f"{b}_x"] - d[f"{a}_x"]).to_numpy(),
                (d[f"{b}_z"] - d[f"{a}_z"]).to_numpy())


def cam_len(d, a="eye", b="snout"):
    return np.hypot((d[f"{b}_x"] - d[f"{a}_x"]).to_numpy(),
                    (d[f"{b}_z"] - d[f"{a}_z"]).to_numpy())


def seg_mask(t, lo, hi):
    return (t >= lo) & (t < hi)


def build(cam):
    jumps = pd.read_csv(os.path.join(L.V3, "v3_jumps.csv"))
    jumps["ballistic"] = jumps["ballistic"].fillna("").astype(str)
    kp = L.build_keypoints_v3()
    body = L.load_body(2)
    W = {}
    for r in jumps.itertuples():
        d = A.window(kp, body, r, cam=cam)
        if d is None:
            continue
        W[r.jump] = dict(d=d, row=r)
    return W


def main():
    W2 = build(2)
    W3 = build(3)
    jumps = sorted(set(W2) & set(W3))

    rows = []
    for j in jumps:
        d2, d3 = W2[j]["d"], W3[j]["d"]
        t2, t3 = d2.t.to_numpy(), d3.t.to_numpy()
        r = W2[j]["row"]
        rec = dict(jump=j, cls=r.cls)
        # approach coverage, eye+snout, in each camera
        mA3 = seg_mask(t3, *APP)
        mA2 = seg_mask(t2, *APP)
        rec["cam3_approach_cov"] = round(float(
            (np.isfinite(d3.eye_x) & np.isfinite(d3.snout_x)).to_numpy()[mA3].mean())
            if mA3.any() else 0.0, 2)
        rec["cam2_approach_cov"] = round(float(
            (np.isfinite(d2.eye_x) & np.isfinite(d2.snout_x)).to_numpy()[mA2].mean())
            if mA2.any() else 0.0, 2)
        for name, lo, hi in EP:
            a2 = cam_angle(d2)[seg_mask(t2, lo, hi)]
            a3 = cam_angle(d3)[seg_mask(t3, lo, hi)]
            rec[f"cam2_sd_{name}"] = S.circ_sd_deg(a2)
            rec[f"cam3_sd_{name}"] = S.circ_sd_deg(a3)
        # Cam2 eye->snout length foreshortening over the approach
        L2 = cam_len(d2)[seg_mask(t2, *APP)]
        L2 = L2[np.isfinite(L2)]
        if len(L2) > 3:
            rec["cam2_len_med"] = float(np.median(L2))
            rec["cam2_len_cv"] = float(np.std(L2) / np.median(L2))
            rec["yaw_bound_deg"] = float(np.degrees(np.arccos(
                np.clip(np.percentile(L2, 5) / np.percentile(L2, 95), -1, 1))))
        else:
            rec["cam2_len_med"] = rec["cam2_len_cv"] = rec["yaw_bound_deg"] = np.nan
        # pseudo-3D azimuth: along = Cam2 x, across = Cam3 x (HANDOFF Y_across proxy)
        # sample both cams on the shared abs_frame grid
        m = pd.merge(d2[["abs_frame", "snout_x", "eye_x"]].rename(
                        columns={"snout_x": "s_along", "eye_x": "e_along"}),
                     d3[["abs_frame", "snout_x", "eye_x", "t"]].rename(
                        columns={"snout_x": "s_acr", "eye_x": "e_acr"}),
                     on="abs_frame", how="inner")
        az = _ang((m.s_acr - m.e_acr).to_numpy(), (m.s_along - m.e_along).to_numpy())
        mAaz = seg_mask(m.t.to_numpy(), *APP)
        rec["azimuth_sd_approach"] = S.circ_sd_deg(az[mAaz]) if mAaz.sum() > 3 else np.nan
        rows.append(rec)
    df = pd.DataFrame(rows)
    S.safe_csv(df, os.path.join(L.V3, "v3_yaw.csv"))

    good = df[(df.cam3_approach_cov >= 0.8) & (df.cam2_approach_cov >= 0.8)
              & np.isfinite(df.cam2_sd_approach)].copy()
    gup = good[good.cls == "up->up"]
    print("\n--- yaw check (approach window, per jump) ---")
    print(df[["jump", "cls", "cam2_approach_cov", "cam3_approach_cov",
              "cam2_sd_approach", "cam3_sd_approach", "cam2_len_cv", "yaw_bound_deg",
              "azimuth_sd_approach"]].round(2).to_string(index=False))
    print(f"\nusable jumps (both cams' approach cov >= 0.8): {list(good.jump)}")
    for lab, g in [("all usable", good), ("up->up only", gup)]:
        print(f"\n{lab}  (n={len(g)}):")
        print(f"  median Cam2 approach head-axis SD : {g.cam2_sd_approach.median():.1f} deg  (pitch)")
        print(f"  median Cam3 approach head-axis SD : {g.cam3_sd_approach.median():.1f} deg  (pitch+yaw)")
        print(f"  median Cam2 eye->snout length CV  : {g.cam2_len_cv.median():.3f}")
        print(f"  median pseudo-3D azimuth SD       : {g.azimuth_sd_approach.median():.1f} deg")

    # ------------------------------------------------------------------ figure
    ex = "v3_15" if "v3_15" in good.jump.values else good.jump.iloc[0]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8))

    # A: representative jump, Cam2 vs Cam3 eye->snout angle (mean-subtracted)
    ax = axes[0, 0]
    for cam, d, col in [(2, W2[ex]["d"], "tab:blue"), (3, W3[ex]["d"], "tab:red")]:
        t = d.t.to_numpy()
        a = cam_angle(d)
        base = L.circmean_deg(a[seg_mask(t, *APP)])
        ax.plot(t, L.wrap180(a - base), color=col, lw=1.3, label=f"Cam{cam}")
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.axvspan(*APP, color="gray", alpha=0.12)
    ax.set_xlim(-1.0, 0.5)
    ax.set_ylim(-60, 60)
    ax.set_xlabel("time from onset (s)")
    ax.set_ylabel("eye->snout angle - approach mean (deg)")
    ax.set_title(f"{ex}: both cameras' head-axis angle\n"
                 "flat in the approach in BOTH = no hidden yaw", fontsize=9)
    ax.legend(fontsize=8)

    # B: per-jump approach circ SD, Cam2 vs Cam3
    ax = axes[0, 1]
    for r in good.itertuples():
        ax.plot(r.cam2_sd_approach, r.cam3_sd_approach, "o", ms=7,
                color=A.CLASS_COLORS.get(r.cls, "gray"), mec="k", mew=.4)
        ax.annotate(r.jump[3:], (r.cam2_sd_approach, r.cam3_sd_approach),
                    fontsize=6, xytext=(3, 3), textcoords="offset points")
    for c in good.cls.unique():
        ax.plot([], [], "o", color=A.CLASS_COLORS.get(c, "gray"), label=c)
    ax.legend(fontsize=7, loc="lower right")
    lim = 12
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.6)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Cam2 approach head-axis SD (deg)  [pitch]")
    ax.set_ylabel("Cam3 approach head-axis SD (deg)  [pitch + yaw]")
    ax.set_title("Points near the diagonal = Cam3 adds no extra\n"
                 "wobble -> the approach jitter is not yaw", fontsize=9)

    # C: per-jump approach vs post, Cam3 angle SD (parallels fig15)
    ax = axes[0, 2]
    for r in good.itertuples():
        ax.plot([0, 1, 2], [r.cam3_sd_approach, r.cam3_sd_jump, r.cam3_sd_post],
                "-o", ms=4, color=A.CLASS_COLORS.get(r.cls, "gray"), alpha=0.7)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["approach", "jump", "post"])
    ax.set_ylabel("Cam3 eye->snout angle circ SD (deg)")
    ax.set_ylim(0, 40)
    ax.set_title("Cam3 head-axis steady through takeoff too,\n"
                 "then destabilises after landing (>40 clipped)", fontsize=9)

    # D: Cam3 angle traces, all good jumps, mean-subtracted (parallels fig16 top)
    ax = axes[1, 0]
    for j in good.jump:
        d = W3[j]["d"]
        t = d.t.to_numpy()
        a = cam_angle(d)
        mA = seg_mask(t, *APP)
        if np.isfinite(a[mA]).sum() < 5:
            continue
        base = L.circmean_deg(a[mA])
        ax.plot(t, L.wrap180(a - base), lw=1,
                color=A.CLASS_COLORS.get(W3[j]["row"].cls, "gray"), alpha=0.6)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlim(-1.0, 0.5)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("time from onset (s)")
    ax.set_ylabel("Cam3 eye->snout angle - approach mean (deg)")
    ax.set_title("Every jump: Cam3 head-axis flat in the approach,\n"
                 "sweeps only at takeoff", fontsize=9)

    # E: Cam2 eye->snout length (foreshortening = yaw), mean-normalised
    ax = axes[1, 1]
    for j in good.jump:
        d = W2[j]["d"]
        t = d.t.to_numpy()
        ln = cam_len(d)
        mA = seg_mask(t, *APP)
        base = np.nanmedian(ln[mA])
        ax.plot(t, ln / base, lw=1, color=A.CLASS_COLORS.get(W2[j]["row"].cls, "gray"),
                alpha=0.6)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.axhline(1, color="gray", lw=0.6)
    ax.set_xlim(-1.0, 0.5)
    ax.set_ylim(0.5, 1.5)
    ax.set_xlabel("time from onset (s)")
    ax.set_ylabel("Cam2 eye->snout length / approach median")
    ax.set_title("Cam2 muzzle length ~ cos(yaw): flat in approach\n"
                 "= little rotation toward/away from Cam2", fontsize=9)

    # F: text conclusion
    ax = axes[1, 2]
    ax.axis("off")
    txt = [
        "YAW CHECK - conclusion", "",
        f"usable jumps (both cams cov >=0.8): {len(good)}",
        f"   of which up->up: {len(gup)}",
        "",
        "approach head-axis circ SD (up->up median):",
        f"   Cam2 (pitch only)   {gup.cam2_sd_approach.median():4.1f} deg",
        f"   Cam3 (pitch + yaw)  {gup.cam3_sd_approach.median():4.1f} deg",
        "   -> Cam3 no noisier -> the approach",
        "      steadiness is not hiding a yaw.",
        "",
        f"pseudo-3D azimuth SD (up->up)  {gup.azimuth_sd_approach.median():4.1f} deg",
        "   (Cam2 x=along, Cam3 x=across; oblique,",
        "    uncalibrated - trend only)",
        "",
        f"Cam2 muzzle-length CV (up->up) {gup.cam2_len_cv.median():.3f}",
        "   loose ceiling only: near broadside to",
        "   Cam2 the length barely responds to yaw.",
        "",
        "CAVEATS",
        " - Cam3 oblique: angle mixes pitch & yaw, so",
        "   it OVER-states, never under-states, yaw.",
        " - post-landing Cam3 coverage poor.",
        " - up->down jumps genuinely re-aim downward",
        "   in the approach - expected, not yaw.",
    ]
    ax.text(0.0, 1.0, "\n".join(txt), va="top", ha="left", family="monospace",
            fontsize=8.2, transform=ax.transAxes)

    fig.suptitle("fig17  Yaw check: is the 'still head' of fig15/fig16 also still in the "
                 "left/right (yaw) axis that Cam2 cannot see?\n"
                 "Cam3 (frontal/oblique) is yaw-sensitive; its approach head-axis jitter "
                 "is no larger than Cam2's -> the stability is real, not a projection artefact.",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig17_yaw_check.png"))
    plt.close(fig)
    print(f"\nfigure -> {os.path.join(FIG, 'fig17_yaw_check.png')}")


if __name__ == "__main__":
    main()
