"""
Step 3: configure SuperAnimal fine-tuning + select the training clips.

- adds a conversion table (project bodyparts -> SuperAnimal-Quadruped) to the
  project config, so create_training_dataset can initialise from SuperAnimal's
  pretrained decoder instead of ImageNet.
- head_top has no SuperAnimal equivalent -> trained from scratch (that's fine).
"""
import os
import deeplabcut
from deeplabcut.modelzoo.utils import create_conversion_table

CFG = open(r"E:\TreeShrew_Jumping\analysis\dlc\PROJECT_CONFIG_PATH.txt").read().strip()

# project bodypart -> SuperAnimal-Quadruped bodypart
CONVERSION = {
    "snout": "nose",
    "eye": "left_eye",
    "ear": "left_earbase",
    "nape": "neck_base",
    # head_top: no SA match, omitted -> random init for that head only
}

# 8 jumps spanning the session x 2 cameras = 16 training clips.
TRAIN_JUMPS = [1, 4, 8, 12, 16, 19, 23, 27]
TRAIN_CLIPS = [f"cam{c}_jump_{j:02d}.mp4" for j in TRAIN_JUMPS for c in (2, 3)]


def main():
    tbl = create_conversion_table(CFG, "superanimal_quadruped", CONVERSION)
    print("conversion table added:", tbl.table if hasattr(tbl, "table") else tbl)

    with open(os.path.join(os.path.dirname(CFG), "TRAIN_CLIPS.txt"), "w") as f:
        f.write("\n".join(TRAIN_CLIPS))
    print(f"{len(TRAIN_CLIPS)} training clips selected:")
    for c in TRAIN_CLIPS:
        print("  ", c)


if __name__ == "__main__":
    main()
