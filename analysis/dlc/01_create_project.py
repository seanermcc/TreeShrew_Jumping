"""
Step 1: create the DeepLabCut project for tree-shrew head tracking.

Bodyparts (head only -- the rest of the animal is covered by the existing
background-subtraction centroid pipeline):
    snout, eye, ear, head_top, nape

Videos: the per-jump Cam2 + Cam3 clips cut by cut_dlc_clips.py
(6 s window, native 1024x1024 @ 100 fps).

Engine: pytorch (DLC 3.x).
"""
import glob
import os

import deeplabcut

ROOT = r"E:\TreeShrew_Jumping\analysis\dlc"
CLIPS = sorted(glob.glob(os.path.join(ROOT, "clips", "cam*_jump_*.mp4")))
BODYPARTS = ["snout", "eye", "ear", "head_top", "nape"]
SKELETON = [
    ["snout", "eye"], ["eye", "ear"], ["ear", "head_top"],
    ["head_top", "nape"], ["eye", "head_top"],
]


def main():
    assert CLIPS, "no clips found -- run cut_dlc_clips.py first"
    config_path = deeplabcut.create_new_project(
        "TreeShrewHead", "sem", CLIPS,
        working_directory=ROOT, copy_videos=False, multianimal=False,
    )
    print("project config:", config_path)

    cfg = deeplabcut.auxiliaryfunctions.read_config(config_path)
    cfg["bodyparts"] = BODYPARTS
    cfg["skeleton"] = SKELETON
    cfg["default_net_type"] = "resnet_50"
    cfg["dotsize"] = 4
    cfg["numframes2pick"] = 6  # ~6 per clip * 54 clips ~= 320 labelled frames
    deeplabcut.auxiliaryfunctions.write_config(config_path, cfg)
    print("configured bodyparts:", cfg["bodyparts"])
    with open(os.path.join(ROOT, "PROJECT_CONFIG_PATH.txt"), "w") as f:
        f.write(config_path)


if __name__ == "__main__":
    main()
