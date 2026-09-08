"""
Cut a synchronized 4-camera review clip for every jump in the inventory.

Window: 3 s before to 3 s after the jump's detection frame (6 s, 600 frames
at 100 Hz). Since all four cameras share frame index 0 with zero lag (see
align.py), the same start time in seconds is used for every camera -- no
per-camera offset needed.

Output: one 2x2 grid video per jump (Cam1 top-left, Cam2 top-right,
Cam3 bottom-left, Cam4 bottom-right), slowed 4x (600 frames stretched over
24 s of output at 25 fps) so a ~150 ms jump is actually visible, with a
label burned into each pane and the jump number/time in a title bar.

ffmpeg's default accurate_seek makes "-ss T -i file" frame-exact when T
precedes -i (verified: a 6.0 s request yields exactly 600 frames).
"""
import os
import subprocess
import sys

import pandas as pd

FFMPEG = os.environ.get("FFMPEG_BIN", "ffmpeg")
FONT = r"C\:/Windows/Fonts/arial.ttf"  # colon escaped for ffmpeg filter syntax

FPS = 100.0
HALF_WINDOW_S = 3.0
DURATION_S = 2 * HALF_WINDOW_S
PANE = 480
SLOWMO = 4          # output plays 4x slower than real time
OUT_FPS = FPS / SLOWMO  # = 25 -> every source frame kept, none dropped


def build_cmd(src_dir, t_start, jump_no, t_jump, out_path):
    inputs = []
    for cam in (1, 2, 3, 4):
        # -ss and -t both precede -i so they trim THIS input's read range;
        # placing -t after all four -i's instead makes it an output-level
        # limit that fights the setpts stretch below.
        inputs += ["-ss", f"{t_start:.3f}", "-t", f"{DURATION_S:.3f}", "-i",
                   os.path.join(src_dir, f"20260730_Cam{cam}.mkv")]

    labels = {1: "Cam1", 2: "Cam2", 3: "Cam3", 4: "Cam4"}
    panes = []
    for k, cam in enumerate((1, 2, 3, 4)):
        panes.append(
            f"[{k}:v]scale={PANE}:{PANE},setpts={SLOWMO}*PTS,"
            f"drawtext=fontfile='{FONT}':text='{labels[cam]}':"
            f"x=8:y=8:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=4"
            f"[v{k}]"
        )
    stack = (
        "[v0][v1][v2][v3]xstack=inputs=4:"
        f"layout=0_0|{PANE}_0|0_{PANE}|{PANE}_{PANE}[grid]"
    )
    title = (
        f"[grid]drawtext=fontfile='{FONT}':"
        f"text='Jump {jump_no:02d}  t={t_jump:.2f}s  (4x slow)':"
        f"x=(w-text_w)/2:y=8:fontsize=30:fontcolor=yellow:box=1:boxcolor=black@0.6:boxborderw=6[out]"
    )
    filt = ";".join(panes) + ";" + stack + ";" + title

    cmd = [FFMPEG, "-v", "error", "-nostdin", *inputs,
           "-filter_complex", filt, "-map", "[out]",
           "-r", f"{OUT_FPS:.3f}",
           "-c:v", "libx264", "-crf", "18", "-preset", "fast",
           "-pix_fmt", "yuv420p", "-y", out_path]
    return cmd


def main():
    src_dir, csv_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    jumps = df[df.is_jump].sort_values("t_s").reset_index(drop=True)
    print(f"{len(jumps)} jumps to cut")

    for i, row in jumps.iterrows():
        jump_no = i + 1
        t_jump = float(row.t_s)
        t_start = max(0.0, t_jump - HALF_WINDOW_S)
        out_path = os.path.join(out_dir, f"jump_{jump_no:02d}.mp4")
        cmd = build_cmd(src_dir, t_start, jump_no, t_jump, out_path)
        print(f"[jump {jump_no:02d}] t={t_jump:7.2f}s  window [{t_start:.2f}, "
              f"{t_start+DURATION_S:.2f}]s -> {out_path}", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  FFMPEG FAILED:\n{r.stderr[-2000:]}")
        else:
            print(f"  ok  ({os.path.getsize(out_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
