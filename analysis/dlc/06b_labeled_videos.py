"""
Step 6b (optional, slow): render labeled overlay videos for visual QC.

~1 min per clip -> ~1 h for all 54. Run 06_analyze.py first.
Pass clip stems to limit it, e.g.:
    python 06b_labeled_videos.py cam2_jump_23 cam3_jump_23
"""
import glob
import os
import sys
import deeplabcut

CFG = open(r"E:\TreeShrew_Jumping\analysis\dlc\PROJECT_CONFIG_PATH.txt").read().strip()
PROJ = os.path.dirname(CFG)


def main():
    if len(sys.argv) > 1:
        vids = [os.path.join(PROJ, "videos", a if a.endswith(".mp4") else a + ".mp4")
                for a in sys.argv[1:]]
    else:
        vids = sorted(glob.glob(os.path.join(PROJ, "videos", "cam*_jump_*.mp4")))
    deeplabcut.create_labeled_video(CFG, vids, shuffle=1, filtered=True,
                                    draw_skeleton=True, pcutoff=0.6)
    print(f"\nrendered {len(vids)} labeled videos -> {PROJ}\\videos\\*_labeled.mp4")


if __name__ == "__main__":
    main()
