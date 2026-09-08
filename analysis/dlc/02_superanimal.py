"""
Step 2: SuperAnimal-Quadruped zero-shot inference on the jump clips.

No training / no labels. Produces per-clip prediction H5/CSV + a labeled
video so we can eyeball how well the pretrained model finds
nose / eyes / earbases on a tree shrew in this arena.

Head keypoints of interest from the 39-point quadruped skeleton:
    nose, left_eye, right_eye, left_earbase, right_earbase, neck_base

Usage:
    python 02_superanimal.py                # all clips
    python 02_superanimal.py cam2_jump_23 cam3_jump_23   # named subset
"""
import glob
import os
import sys

import deeplabcut

ROOT = r"E:\TreeShrew_Jumping\analysis\dlc"
CLIP_DIR = os.path.join(ROOT, "clips")
OUT = os.path.join(ROOT, "superanimal")


def main():
    os.makedirs(OUT, exist_ok=True)
    if len(sys.argv) > 1:
        vids = [os.path.join(CLIP_DIR, a if a.endswith(".mp4") else a + ".mp4")
                for a in sys.argv[1:]]
    else:
        vids = sorted(glob.glob(os.path.join(CLIP_DIR, "cam*_jump_*.mp4")))
    print(f"{len(vids)} clips")

    deeplabcut.video_inference_superanimal(
        vids,
        superanimal_name="superanimal_quadruped",
        model_name="hrnet_w32",
        detector_name="fasterrcnn_resnet50_fpn_v2",
        dest_folder=OUT,
        max_individuals=1,
        batch_size=8,
        detector_batch_size=4,
        pcutoff=0.2,
        create_labeled_video=True,
        plot_bboxes=True,
        device="cuda",
    )
    print("done ->", OUT)


if __name__ == "__main__":
    main()
