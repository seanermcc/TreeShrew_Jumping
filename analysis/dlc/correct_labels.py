"""
Lightweight keypoint correction GUI (matplotlib/Tk -- no OpenGL, no napari).

Loads every labeled-data folder that has a CollectedData_sem.h5, shows each
frame with its 5 head points, and lets you drag them into place. Writes the
same DeepLabCut CollectedData_sem.h5 / .csv that training expects.

Controls
  drag a dot            move that point
  1 2 3 4 5             arm snout / eye / ear / head_top / nape, then click to place
  d  or  Delete         remove the point nearest the cursor  (marks it absent)
  n / p   or  -> / <-   next / previous frame   (n past the last frame -> next clip)
  ]  /  [               NEXT / PREVIOUS CLIP   (16 clips total - do them all!)
  scroll wheel          zoom under the cursor
  0                     reset zoom      f  fit to the animal
  s                     save now   (also auto-saves on every move between frames/clips)
  q  or close window    save and quit
  h                     toggle this help
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ = os.path.dirname(
    open(r"E:\TreeShrew_Jumping\analysis\dlc\PROJECT_CONFIG_PATH.txt").read().strip())
LD = os.path.join(PROJ, "labeled-data")
SCORER = "sem"
BP = ["snout", "eye", "ear", "head_top", "nape"]
COLORS = {"snout": "#ff3b3b", "eye": "#35c73b", "ear": "#2f8bff",
          "head_top": "#ffd21e", "nape": "#e64bd0"}
PICK_RADIUS = 24


class Clip:
    def __init__(self, folder):
        self.folder = folder
        self.name = os.path.basename(folder)
        self.h5 = os.path.join(folder, f"CollectedData_{SCORER}.h5")
        self.df = pd.read_hdf(self.h5)
        self.scorer = self.df.columns.get_level_values(0)[0]
        self.imgs = [ix[-1] for ix in self.df.index]
        self.dirty = False

    def img_path(self, i):
        return os.path.join(self.folder, self.imgs[i])

    def get(self, i, bp):
        r = self.df.iloc[i]
        return float(r[(self.scorer, bp, "x")]), float(r[(self.scorer, bp, "y")])

    def set(self, i, bp, xy):
        k = self.df.index[i]
        self.df.loc[k, (self.scorer, bp, "x")] = xy[0] if xy else np.nan
        self.df.loc[k, (self.scorer, bp, "y")] = xy[1] if xy else np.nan
        self.dirty = True

    def save(self):
        if not self.dirty:
            return
        self.df.to_hdf(self.h5, key="df_with_missing", mode="w")
        self.df.to_csv(os.path.join(self.folder, f"CollectedData_{SCORER}.csv"))
        self.dirty = False


class Labeler:
    def __init__(self, clips):
        self.clips, self.ci, self.fi = clips, 0, 0
        self.armed = self.drag = None
        self.show_help = True
        self.seen = {0}
        self.fig, self.ax = plt.subplots(figsize=(11, 9))
        self.fig.subplots_adjust(0, 0, 1, 0.94)
        self._im = self.ax.imshow(plt.imread(self.clip.img_path(0)), cmap="gray")
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.pts = {b: self.ax.plot([], [], "o", ms=11, mfc=COLORS[b],
                                    mec="white", mew=1.6, zorder=5)[0] for b in BP}
        self.labels = {b: self.ax.annotate(b, (0, 0), (7, 7),
                       textcoords="offset points", color=COLORS[b],
                       fontsize=9, weight="bold", zorder=6) for b in BP}
        self.helptext = self.ax.text(
            0.005, 0.995, __doc__.split("Controls")[1], transform=self.ax.transAxes,
            va="top", fontsize=8, family="monospace", color="white", zorder=10,
            bbox=dict(boxstyle="round", fc="black", alpha=0.7))
        legend = [
            ("snout", "nose tip"), ("eye", "centre of eyeball"),
            ("ear", "base of ear at skull"), ("head_top", "crown of skull"),
            ("nape", "skull base / neck"),
        ]
        for j, (b, where) in enumerate(legend):
            self.ax.text(0.995, 0.995 - j * 0.032, f"● {b} - {where}",
                         transform=self.ax.transAxes, ha="right", va="top",
                         fontsize=8.5, weight="bold", color=COLORS[b], zorder=10,
                         bbox=dict(boxstyle="round", fc="black", alpha=0.6))
        for ev, fn in [("button_press_event", self.on_press),
                       ("motion_notify_event", self.on_motion),
                       ("button_release_event", self.on_release),
                       ("key_press_event", self.on_key),
                       ("scroll_event", self.on_scroll),
                       ("close_event", lambda e: self.save_current())]:
            self.fig.canvas.mpl_connect(ev, fn)
        self.load_frame(reset=True)

    @property
    def clip(self):
        return self.clips[self.ci]

    def load_frame(self, reset=False):
        c = self.clip
        img = plt.imread(c.img_path(self.fi))
        self._im.set_data(img)
        self._im.set_extent((-0.5, img.shape[1] - 0.5, img.shape[0] - 0.5, -0.5))
        if reset:
            self.ax.set_xlim(-0.5, img.shape[1] - 0.5)
            self.ax.set_ylim(img.shape[0] - 0.5, -0.5)
        self.refresh_points()

    def refresh_points(self):
        c = self.clip
        for b in BP:
            x, y = c.get(self.fi, b)
            if np.isfinite(x):
                self.pts[b].set_data([x], [y])
                self.labels[b].set_position((x, y)); self.labels[b].set_visible(True)
            else:
                self.pts[b].set_data([], [])
                self.labels[b].set_visible(False)
        miss = [b for b in BP if not np.isfinite(c.get(self.fi, b)[0])]
        self.fig.suptitle(
            f"CLIP {self.ci+1} of {len(self.clips)}    ] / [  = next / prev clip"
            f"     (visited {len(self.seen)}/{len(self.clips)})\n"
            f"{c.name}   frame {self.fi+1}/{len(c.imgs)}   (n/p = next/prev frame)"
            f"{'   *UNSAVED*' if c.dirty else ''}\n"
            f"missing on this frame: {', '.join(miss) or 'none'}"
            f"{'     ARMED: ' + self.armed if self.armed else ''}", fontsize=10)
        self.helptext.set_visible(self.show_help)
        self.fig.canvas.draw_idle()

    def nearest(self, x, y):
        if x is None:
            return None
        best, bd = None, PICK_RADIUS ** 2
        # scale pick radius to current zoom
        x0, x1 = self.ax.get_xlim()
        sc = (x1 - x0) / self._im.get_array().shape[1]
        bd *= max(sc, 0.35) ** 2
        for b in BP:
            px, py = self.clip.get(self.fi, b)
            if np.isfinite(px):
                d = (px - x) ** 2 + (py - y) ** 2
                if d < bd:
                    best, bd = b, d
        return best

    def on_press(self, e):
        if e.inaxes != self.ax or e.xdata is None:
            return
        if self.armed:
            self.clip.set(self.fi, self.armed, (e.xdata, e.ydata))
            self.armed = None
            self.refresh_points()
        elif (b := self.nearest(e.xdata, e.ydata)):
            self.drag = b

    def on_motion(self, e):
        if self.drag and e.inaxes == self.ax and e.xdata is not None:
            self.clip.set(self.fi, self.drag, (e.xdata, e.ydata))
            self.pts[self.drag].set_data([e.xdata], [e.ydata])
            self.labels[self.drag].set_position((e.xdata, e.ydata))
            self.fig.canvas.draw_idle()

    def on_release(self, e):
        if self.drag:
            self.drag = None
            self.refresh_points()

    def on_scroll(self, e):
        if e.inaxes != self.ax or e.xdata is None:
            return
        f = 0.8 if e.button == "up" else 1.25
        x0, x1 = self.ax.get_xlim(); y1, y0 = self.ax.get_ylim()
        self.ax.set_xlim(e.xdata - (e.xdata - x0) * f, e.xdata + (x1 - e.xdata) * f)
        self.ax.set_ylim(e.ydata + (y1 - e.ydata) * f, e.ydata - (e.ydata - y0) * f)
        self.fig.canvas.draw_idle()

    def save_current(self):
        self.clip.save()

    def go_frame(self, d):
        self.save_current()
        self.fi = max(0, min(len(self.clip.imgs) - 1, self.fi + d))
        self.armed = None
        self.load_frame()

    def go_clip(self, d):
        self.save_current()
        self.ci = max(0, min(len(self.clips) - 1, self.ci + d))
        self.fi = 0
        self.armed = None
        self.seen.add(self.ci)
        self.load_frame(reset=True)

    def fit_animal(self):
        xs = [self.clip.get(self.fi, b) for b in BP]
        xs = [(x, y) for x, y in xs if np.isfinite(x)]
        if not xs:
            return
        cx = np.mean([p[0] for p in xs]); cy = np.mean([p[1] for p in xs])
        self.ax.set_xlim(cx - 210, cx + 210); self.ax.set_ylim(cy + 170, cy - 170)
        self.fig.canvas.draw_idle()

    def on_key(self, e):
        k = e.key
        if k in "12345":
            self.armed = BP[int(k) - 1]; self.refresh_points()
        elif k in ("d", "delete", "backspace"):
            if (b := self.nearest(e.xdata, e.ydata) or self.armed):
                self.clip.set(self.fi, b, None); self.armed = None; self.refresh_points()
        elif k in ("n", "right"):
            if self.fi == len(self.clip.imgs) - 1 and self.ci < len(self.clips) - 1:
                self.go_clip(1)          # roll over to next clip
            else:
                self.go_frame(1)
        elif k in ("p", "left"):
            self.go_frame(-1)
        elif k in ("N", "]", "pagedown"):
            self.go_clip(1)
        elif k in ("P", "[", "pageup"):
            self.go_clip(-1)
        elif k == "0":
            self.load_frame(reset=True)
        elif k == "f":
            self.fit_animal()
        elif k == "s":
            self.save_current(); self.refresh_points()
        elif k == "h":
            self.show_help = not self.show_help; self.refresh_points()
        elif k == "q":
            self.save_current(); plt.close(self.fig)


def main():
    folders = sorted(d for d in glob.glob(os.path.join(LD, "*"))
                     if os.path.exists(os.path.join(d, f"CollectedData_{SCORER}.h5")))
    if not folders:
        print("no labeled-data folders with CollectedData_sem.h5 found")
        sys.exit(1)
    clips = [Clip(f) for f in folders]
    print(f"{len(clips)} clips, {sum(len(c.imgs) for c in clips)} frames")
    Labeler(clips)
    plt.show()
    for c in clips:
        c.save()
    print("saved.")


if __name__ == "__main__":
    main()
