"""
Regenerate the head-orientation and head-axis-stability figures on the v4 jump
set: fig06 (v3_analysis) and fig15 / fig16 / fig16b (v3_stability).

v4 = v3 with v3_05 / v3_07 dropped (not jumps) and v3_06 / v3_12 re-timed -- see
v4/README.md and v4_build.py. Jump IDs are unchanged. The plotting code is NOT
modified: this driver just feeds v4/v4_jumps.csv + v4/_keypoints_v4.csv into the
existing figure functions instead of the v3 tables, and writes the PNGs (plus
the recomputed stability / fixation CSVs) into v4/.

    E:\\TreeShrew_Jumping\\analysis\\dlc\\py310\\python.exe v4_figures.py

Outputs (all under v4/):
    figures/fig06_head_orientation.png
    figures/fig15_head_axis_stability.png
    figures/fig16_fixation_point.png
    figures/fig16b_fixation_worked_example.png
    v4_stability.csv          (per-jump within-trial head-axis jitter, fig15)
    v4_fixation_point.csv     (per-jump fixation-point spread, fig16)

Only these four figures were asked for. The same pattern extends to the rest of
the v3 figure set (fig01-09c in v3_analysis, fig10-14 in v3_extra) if wanted.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import v3_lib as L
import v3_analysis as A
import v3_stability as S

V4 = os.path.join(L.AN, "v4")
FIG4 = os.path.join(V4, "figures")
os.makedirs(FIG4, exist_ok=True)

# every figure function saves to its module's FIG global -- point them at v4/
A.FIG = FIG4
S.FIG = FIG4


def load_v4():
    jumps = pd.read_csv(os.path.join(V4, "v4_jumps.csv"))
    jumps["ballistic"] = jumps["ballistic"].fillna("").astype(str)
    kp = pd.read_csv(os.path.join(V4, "_keypoints_v4.csv"))
    return jumps, kp


def build_W(jumps, kp):
    """Per-jump -1..+3 s window assembly -- identical to v3_analysis.main(),
    but on the v4 tables (new onsets already baked into v4_jumps.onset_frame)."""
    body = L.load_body(2)
    paws = A.load_paw_contacts()
    W = {}
    for r in jumps.itertuples():
        d = A.window(kp, body, r, cam=2)
        if d is None:
            print(f"  !! no Cam2 window for {r.jump}")
            continue
        sgn = A.travel_sign(d)
        dm = A.mirror(d, sgn)
        xt, zt, t_land = A.landing_target(dm, r)
        gap = A.crosses_gap(d)
        if r.jump in paws:
            px, pz, pconf = paws[r.jump]
            paw_raw, paw = (px, pz), (-px if sgn < 0 else px, pz)
        else:
            paw_raw = paw = (np.nan, np.nan)
            pconf = ""
        W[r.jump] = dict(raw=d, m=dm, sgn=sgn, target=(xt, zt), t_land=t_land,
                         row=r, gap=gap,
                         edge=A.edge_target(sgn) if gap else (np.nan, np.nan),
                         paw=paw, paw_raw=paw_raw, paw_conf=pconf)
    return W


def main():
    jumps, kp = load_v4()
    grid = np.arange(-A.PRE_S, A.POST_S + 1e-9, 0.01)

    W = build_W(jumps, kp)
    print(f"{len(W)} jumps: " + ", ".join(f"{j}[{W[j]['row'].cls}]" for j in W))

    by_cls = {c: [j for j in W if W[j]["row"].cls == c] for c in A.CLASS_COLORS}
    by_cls = {c: v for c, v in by_cls.items() if v}
    print("by class: " + ", ".join(f"{c} n={len(v)}" for c, v in by_cls.items()))

    def stack(col, key="raw", subset=None):
        ids = [j for j in W if subset is None or j in subset]
        return np.vstack([A.resample(W[j][key], col, grid) for j in ids]), ids

    # ---- fig06 : head orientation vs travel axis ---------------------------- #
    A.fig_orientation(grid, stack, by_cls)
    print("  wrote fig06_head_orientation.png")

    # ---- fig15 / fig16 / fig16b : head-axis stability --------------------- #
    df = S.per_jump_table(W)
    S.safe_csv(df, os.path.join(V4, "v4_stability.csv"))
    ex = S._pick_example(W)

    S.fig_stability(df, W)
    print("  wrote fig15_head_axis_stability.png")

    S.fig_fixation_worked_example(W, ex)
    print(f"  wrote fig16b_fixation_worked_example.png  (worked example: {ex})")

    pf = S.fig_fixation(W, grid)
    if pf is not None and len(pf):
        S.safe_csv(pf, os.path.join(V4, "v4_fixation_point.csv"))
    print("  wrote fig16_fixation_point.png")

    # ---- console summary (mirrors v3_stability.main) ---------------------- #
    show = ["jump", "cls", "ballistic", "sd_approach", "sd_jump", "sd_post",
            "sddet_approach", "pkav_approach", "ang_range_approach",
            "bearing_sd_approach"]
    print("\n--- within-trial head-axis stability (per jump, v4) ---")
    print(df[show].round(1).to_string(index=False))
    print("\nmedian across jumps:")
    for n, _, _ in S.EP:
        print(f"  {n:>9}: circ SD {df[f'sd_{n}'].median():5.1f} deg   "
              f"detrended {df[f'sddet_{n}'].median():4.1f} deg   "
              f"peak|w| {df[f'pkav_{n}'].median():6.0f} deg/s")
    up = df[df.cls == "up->up"]
    print(f"\nup->up approach: head-axis angle 5-95 range = "
          f"{up['ang_range_approach'].median():.1f} deg (median), "
          f"axis->edge bearing SD = {up['bearing_sd_approach'].median():.1f} deg")
    n_lo = int((df[["sd_approach", "sd_jump"]].max(axis=1) < df.sd_post).sum())
    print(f"{n_lo}/{df.sd_post.notna().sum()} jumps: both approach and jump-epoch "
          f"SD below the post-landing SD")
    if pf is not None and len(pf):
        print("\n--- fixation-point spread (5-95 px range, v4) ---")
        print(pf.round(1).to_string(index=False))
    print(f"\nfigures + CSVs -> {V4}")


if __name__ == "__main__":
    main()
