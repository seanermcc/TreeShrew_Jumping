"""
Forepaw touchdown labeller (matplotlib / Tk -- no OpenGL, no napari).

One jump at a time:

  LEFT   a contact sheet of Cam2 frames, jump onset -0.05 s .. +0.55 s.
         Click the frame where the forepaw actually plants -> it loads on the
         right as the "adjusted image" and becomes this jump's contact frame.

  RIGHT  the chosen frame, zoomed on the landing zone. Click (or drag) to put
         the target where the forepaw actually lands. The head keypoints, the
         head->target line and the resulting aiming angle are drawn live.

Everything autosaves to  v4/paw_contacts_v4.csv  (seeded from the old
v3/paw_contacts.csv). Close the window when done.

Controls
  click a left thumbnail      choose that frame  (sets the contact frame)
  click / drag on the right   move the touchdown target
  , / .                       nudge the chosen frame -1 / +1
  [ / ]   or  - / +           zoom the right panel out / in
  scroll wheel (right panel)  zoom under the cursor
  n / p   or  Right / Left    next / previous jump
  c                           cycle confidence  high -> med -> low
  s                           toggle surface    platform <-> floor
  x                           toggle "no discrete contact" flag for this jump
  S                           save now (also autosaves on every change)
  q  / close window           save and quit

    E:\\TreeShrew_Jumping\\analysis\\dlc\\py310\\python.exe v4_paw_gui.py  [v3_04 v3_08 ...]
"""
import os
import sys
import subprocess

import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams["toolbar"] = "None"      # clicks place the target, not pan/zoom
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import v3_lib as L

V4 = os.path.join(L.AN, "v4")
OUT = os.path.join(V4, "paw_contacts_v4.csv")
SEED = os.path.join(L.V3, "paw_contacts.csv")

PRE_FR, POST_FR = -5, 55        # contact-sheet window, frames from onset
GRID_ROWS, GRID_COLS = 5, 6     # 30 thumbnails (every 2nd frame)
CACHE_PRE, CACHE_POST = -12, 62
CONF_CYCLE = ["high", "med", "low"]


# --------------------------------------------------------------------------- #
def grab_range(cam, f_start, n):
    """n consecutive Cam frames as a (n, H, W) uint8 stack, f_start = frame of
    stack[0]. `-ss T -i` is frame-exact with the bundled ffmpeg (see
    cut_jump_clips.py)."""
    t = f_start / L.FPS
    cmd = [L.FFMPEG, "-v", "error", "-nostdin", "-ss", f"{t:.3f}",
           "-i", L.SRC.format(cam), "-frames:v", str(n),
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    buf = subprocess.run(cmd, capture_output=True).stdout
    px = L.IMG * L.IMG
    nn = len(buf) // px
    return np.frombuffer(buf[:nn * px], np.uint8).reshape(nn, L.IMG, L.IMG)


def ang_deg(dx, dy_up):
    return float(np.degrees(np.arctan2(dy_up, dx)))


# --------------------------------------------------------------------------- #
class Store:
    """Per-jump touchdown record, backed by paw_contacts_v4.csv."""

    COLS = ["jump", "contact_frame", "paw_x_img", "paw_y_img", "surface",
            "confidence", "note", "onset_frame", "dframe", "status", "edited"]

    def __init__(self, jumps):
        self.jumps = jumps
        seed = pd.read_csv(SEED).set_index("jump") if os.path.exists(SEED) else pd.DataFrame()
        prev = pd.read_csv(OUT).set_index("jump") if os.path.exists(OUT) else pd.DataFrame()
        # rows already in the output file for jumps NOT in this session -> keep as-is
        inscope = set(jumps.jump)
        self.passthrough = (prev[~prev.index.isin(inscope)].reset_index()
                            if len(prev) else pd.DataFrame(columns=self.COLS))
        self.rec = {}
        for r in jumps.itertuples():
            o = int(r.onset_frame)
            src = prev if r.jump in prev.index else seed
            if r.jump in src.index:
                s = src.loc[r.jump]
                cf = int(s.get("contact_frame", o + 8) or o + 8)
                d = dict(contact_frame=cf,
                         paw_x_img=float(s.paw_x_img), paw_y_img=float(s.paw_y_img),
                         surface=str(s.get("surface", "platform")),
                         confidence=str(s.get("confidence", "med")),
                         note=str(s.get("note", "")),
                         status=str(s.get("status", "ok")) if r.jump in prev.index else "ok",
                         edited=bool(s.get("edited", False)) if r.jump in prev.index else False)
            else:                                   # v3_01 / v3_14 -- no seed
                d = dict(contact_frame=o + 8, paw_x_img=L.IMG / 2, paw_y_img=L.IMG / 2,
                         surface="platform", confidence="low", note="", status="none",
                         edited=False)
            d["onset_frame"] = o
            self.rec[r.jump] = d

    def save(self):
        rows = []
        for j, d in self.rec.items():
            rows.append(dict(jump=j, contact_frame=d["contact_frame"],
                             paw_x_img=round(d["paw_x_img"], 1),
                             paw_y_img=round(d["paw_y_img"], 1),
                             surface=d["surface"], confidence=d["confidence"],
                             note=d["note"], onset_frame=d["onset_frame"],
                             dframe=d["contact_frame"] - d["onset_frame"],
                             status=d["status"], edited=d["edited"]))
        df = pd.DataFrame(rows)
        if len(self.passthrough):
            df = pd.concat([df, self.passthrough[self.COLS]], ignore_index=True)
        order = {j: i for i, j in enumerate(self.jumps.jump)}
        df["_o"] = df.jump.map(lambda j: order.get(j, 999))
        df = df.sort_values("_o").drop(columns="_o")
        df[self.COLS].to_csv(OUT, index=False)


# --------------------------------------------------------------------------- #
class Labeller:
    def __init__(self, jumps, kp):
        self.jumps = jumps.reset_index(drop=True)
        self.kp = kp
        self.store = Store(self.jumps)
        self.ji = 0
        self.half = 190                       # right-panel crop half-size (px)
        self.stack = None
        self.stack_f0 = 0
        self.detail_frame = 0
        self.placing = False

        self.fig = plt.figure(figsize=(17.5, 9.2))
        gs = self.fig.add_gridspec(GRID_ROWS, GRID_COLS + 4, wspace=0.04, hspace=0.22,
                                   left=0.03, right=0.985, top=0.9, bottom=0.06)
        self.thumb_ax, self.thumb_frame = [], []
        for rr in range(GRID_ROWS):
            for cc in range(GRID_COLS):
                self.thumb_ax.append(self.fig.add_subplot(gs[rr, cc]))
        self.detail_ax = self.fig.add_subplot(gs[:, GRID_COLS:])
        for a in (*self.thumb_ax, self.detail_ax):
            a.set_xticks([]); a.set_yticks([])

        for ev, fn in [("button_press_event", self.on_press),
                       ("motion_notify_event", self.on_motion),
                       ("button_release_event", self.on_release),
                       ("key_press_event", self.on_key),
                       ("scroll_event", self.on_scroll),
                       ("close_event", lambda e: self.store.save())]:
            self.fig.canvas.mpl_connect(ev, fn)

        self.load_jump()

    # --- current-jump helpers ------------------------------------------------ #
    @property
    def row(self):
        return self.jumps.iloc[self.ji]

    @property
    def d(self):
        return self.store.rec[self.row.jump]

    def head_at(self, frame):
        k = self.kp[(self.kp.cam == 2) & (self.kp.abs_frame == frame)]
        return k.iloc[0] if len(k) else None

    def crop_center(self):
        return self.d["paw_x_img"], self.d["paw_y_img"]

    # --- loading ----------------------------------------------------------- #
    def load_jump(self):
        r = self.row
        o = int(r.onset_frame)
        self.fig.suptitle(f"loading {r.jump} ...", fontsize=12)
        self.fig.canvas.draw_idle(); self.fig.canvas.flush_events()
        self.stack_f0 = o + CACHE_PRE
        self.stack = grab_range(2, self.stack_f0, CACHE_POST - CACHE_PRE)
        self.detail_frame = int(np.clip(self.d["contact_frame"],
                                        o + PRE_FR, o + POST_FR))
        self.draw_thumbs()
        self.draw_detail(recenter=True)
        self.refresh_title()

    def frame_img(self, frame):
        i = frame - self.stack_f0
        if 0 <= i < len(self.stack):
            return self.stack[i]
        return np.zeros((L.IMG, L.IMG), np.uint8)

    # --- drawing --------------------------------------------------------- #
    def draw_thumbs(self):
        r = self.row
        o = int(r.onset_frame)
        px, py = self.d["paw_x_img"], self.d["paw_y_img"]
        surf = self.d["surface"]
        cw, ch = (560, 440) if surf == "floor" else (470, 340)
        cx = px
        cy = py - 80 if surf == "floor" else py - 20
        x0 = int(np.clip(cx - cw / 2, 0, L.IMG - cw))
        y0 = int(np.clip(cy - ch / 2, 0, L.IMG - ch))
        frames = np.linspace(o + PRE_FR, o + POST_FR, len(self.thumb_ax)).round().astype(int)
        self.thumb_frame = list(frames)
        for ax, fr in zip(self.thumb_ax, frames):
            ax.clear(); ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(self.frame_img(fr)[y0:y0 + ch, x0:x0 + cw], cmap="gray",
                      vmin=0, vmax=255, extent=[x0, x0 + cw, y0 + ch, y0])
            ax.plot(px, py, "+", ms=11, mew=1.5, color="cyan")
            ax.set_xlim(x0, x0 + cw); ax.set_ylim(y0 + ch, y0)
            sel = fr == self.detail_frame
            for s in ax.spines.values():
                s.set_color("yellow" if sel else "0.6")
                s.set_linewidth(2.4 if sel else 0.8)
            ax.set_title(f"{(fr - o) / L.FPS * 1000:+.0f} ms", fontsize=7.5,
                         color="yellow" if sel else "black")

    def draw_detail(self, recenter=False):
        ax = self.detail_ax
        ax.clear(); ax.set_xticks([]); ax.set_yticks([])
        o = int(self.row.onset_frame)
        fr = self.detail_frame
        px, py = self.d["paw_x_img"], self.d["paw_y_img"]
        row = self.head_at(fr)
        if recenter or not hasattr(self, "_cxy"):
            # centre on the head if we have it this frame, else on the target
            if row is not None and np.isfinite(row["snout_x"]):
                self._cxy = (0.5 * (float(row["snout_x"]) + px),
                             0.5 * (float(row["snout_y"]) + py))
            else:
                self._cxy = (px, py)
        cx, cy = self._cxy
        h = self.half
        x0, x1 = cx - h, cx + h
        y0, y1 = cy - h, cy + h
        ax.imshow(self.frame_img(fr), cmap="gray", vmin=0, vmax=255)
        # head keypoints for this frame
        aim = None
        if row is not None:
            for bp, col, ms in [("snout", "red", 7), ("eye", "yellow", 5),
                                ("ear", "lime", 5), ("head_top", "cyan", 5),
                                ("nape", "magenta", 7)]:
                if np.isfinite(row[f"{bp}_x"]):
                    ax.plot(row[f"{bp}_x"], row[f"{bp}_y"], "o", ms=ms, color=col,
                            mec="k", mew=.5)
            if np.isfinite(row["nape_x"]) and np.isfinite(row["snout_x"]):
                axis = ang_deg(row["snout_x"] - row["nape_x"],
                               -(row["snout_y"] - row["nape_y"]))
                bear = ang_deg(px - row["nape_x"], -(py - row["nape_y"]))
                aim = (axis - bear + 180) % 360 - 180
                reach = float(np.hypot(px - row["nape_x"], py - row["nape_y"]))
                ex = row["nape_x"] + reach * np.cos(np.deg2rad(axis))
                ey = row["nape_y"] - reach * np.sin(np.deg2rad(axis))
                ax.plot([row["nape_x"], ex], [row["nape_y"], ey], "--", color="tab:blue", lw=1.4)
                ax.plot([row["nape_x"], px], [row["nape_y"], py], "-", color="lime", lw=1.4)
        # the target
        ax.plot(px, py, "+", ms=26, mew=2.6, color="cyan")
        ax.plot(px, py, "o", ms=24, mfc="none", mec="cyan", mew=1.6)
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
        ttl = (f"ADJUSTED IMAGE   f{fr}   ({(fr - o) / L.FPS * 1000:+.0f} ms)"
               f"   target ({px:.0f}, {py:.0f})")
        if aim is not None:
            ttl += f"   aim err {aim:+.0f}\u00b0"
        ax.set_title(ttl, fontsize=10)

    def refresh_title(self):
        r = self.row
        d = self.d
        done = sum(1 for v in self.store.rec.values() if v["edited"])
        self.fig.suptitle(
            f"jump {self.ji + 1}/{len(self.jumps)}:  {r.jump}  [{r.cls}]"
            f"      surface={d['surface']}   conf={d['confidence']}"
            f"   status={d['status']}   {'EDITED' if d['edited'] else '(unedited)'}"
            f"      edited {done}/{len(self.jumps)}\n"
            f"click the plant frame on the left  \u2192  click/drag the target on the "
            f"right   |   , . frame   [ ] zoom   n/p jump   c conf   s surface   "
            f"x no-contact   q quit", fontsize=9.5)
        self.fig.canvas.draw_idle()

    # --- events --------------------------------------------------------- #
    def on_press(self, e):
        if e.inaxes in self.thumb_ax and e.xdata is not None:
            self.detail_frame = self.thumb_frame[self.thumb_ax.index(e.inaxes)]
            self.d["contact_frame"] = self.detail_frame
            self.d["edited"] = True
            self.draw_thumbs(); self.draw_detail(recenter=True)
            self.refresh_title(); self.store.save()
        elif e.inaxes is self.detail_ax and e.xdata is not None:
            self.placing = True
            self._set_target(e.xdata, e.ydata)

    def on_motion(self, e):
        if self.placing and e.inaxes is self.detail_ax and e.xdata is not None:
            self._set_target(e.xdata, e.ydata, light=True)

    def on_release(self, e):
        if self.placing:
            self.placing = False
            self.draw_thumbs()
            self.refresh_title()
            self.store.save()

    def _set_target(self, x, y, light=False):
        self.d["paw_x_img"] = float(np.clip(x, 0, L.IMG - 1))
        self.d["paw_y_img"] = float(np.clip(y, 0, L.IMG - 1))
        self.d["edited"] = True
        if self.d["status"] == "none":
            self.d["status"] = "ok"
        self.draw_detail()

    def on_scroll(self, e):
        if e.inaxes is self.detail_ax and e.xdata is not None:
            self._cxy = (e.xdata, e.ydata)
            self.half = int(np.clip(self.half * (0.8 if e.button == "up" else 1.25),
                                    50, 512))
            self.draw_detail()
            self.fig.canvas.draw_idle()

    def go_jump(self, step):
        self.store.save()
        self.ji = int(np.clip(self.ji + step, 0, len(self.jumps) - 1))
        self.load_jump()

    def nudge_frame(self, step):
        o = int(self.row.onset_frame)
        self.detail_frame = int(np.clip(self.detail_frame + step,
                                        self.stack_f0, self.stack_f0 + len(self.stack) - 1))
        self.d["contact_frame"] = self.detail_frame
        self.d["edited"] = True
        self.draw_thumbs(); self.draw_detail()
        self.refresh_title(); self.store.save()

    def on_key(self, e):
        k = e.key
        if k in ("n", "right"):
            self.go_jump(1)
        elif k in ("p", "left"):
            self.go_jump(-1)
        elif k == ",":
            self.nudge_frame(-1)
        elif k == ".":
            self.nudge_frame(1)
        elif k in ("[", "-"):
            self.half = int(np.clip(self.half * 1.25, 50, 512)); self.draw_detail(); self.fig.canvas.draw_idle()
        elif k in ("]", "+", "="):
            self.half = int(np.clip(self.half * 0.8, 50, 512)); self.draw_detail(); self.fig.canvas.draw_idle()
        elif k == "c":
            i = CONF_CYCLE.index(self.d["confidence"]) if self.d["confidence"] in CONF_CYCLE else 1
            self.d["confidence"] = CONF_CYCLE[(i + 1) % 3]
            self.refresh_title(); self.store.save()
        elif k == "s":
            self.d["surface"] = "floor" if self.d["surface"] == "platform" else "platform"
            self.draw_thumbs(); self.refresh_title(); self.store.save()
        elif k == "x":
            self.d["status"] = "ok" if self.d["status"] == "no_contact" else "no_contact"
            self.d["edited"] = True
            self.refresh_title(); self.store.save()
        elif k == "S":
            self.store.save(); print(f"saved -> {OUT}")
        elif k == "q":
            self.store.save(); plt.close(self.fig)


# --------------------------------------------------------------------------- #
def main():
    jumps = pd.read_csv(os.path.join(V4, "v4_jumps.csv"))
    if len(sys.argv) > 1:
        jumps = jumps[jumps.jump.isin(sys.argv[1:])]
    jumps = jumps.reset_index(drop=True)
    kp = pd.read_csv(os.path.join(V4, "_keypoints_v4.csv"))
    print(f"{len(jumps)} jumps -> labelling. output: {OUT}")
    Labeller(jumps, kp)
    plt.show()
    print("done.")


if __name__ == "__main__":
    main()
