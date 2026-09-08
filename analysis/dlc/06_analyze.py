"""
Step 6: run the trained model on all 54 jump clips (fast: no video rendering).

  analyze_videos     -> raw predictions   (skips clips already done)
  filterpredictions  -> median-filtered predictions (used by 07_head_pose.py)

Outputs land in TreeShrewHead.../videos/ as *DLC*.h5 / *_filtered.h5 (+ .csv).
Then 07_head_pose.py turns them into head position + angle in absolute time.

Labeled overlay videos are rendered separately by 06b_labeled_videos.py
(that step is slow -- ~1 min per clip).
"""
import glob
import os
import deeplabcut

CFG = open(r"E:\TreeShrew_Jumping\analysis\dlc\PROJECT_CONFIG_PATH.txt").read().strip()
PROJ = os.path.dirname(CFG)
VIDEOS = sorted(glob.glob(os.path.join(PROJ, "videos", "cam*_jump_*.mp4")))


def main():
    deeplabcut.analyze_videos(CFG, VIDEOS, shuffle=1, device="cuda",
                              batch_size=16, save_as_csv=True)
    deeplabcut.filterpredictions(CFG, VIDEOS, shuffle=1,
                                 filtertype="median", windowlength=5,
                                 save_as_csv=True)
    print(f"\nanalyzed + filtered {len(VIDEOS)} clips -- now run 07_head_pose.py")


if __name__ == "__main__":
    main()
