"""
Step 5: build the training dataset (SuperAnimal-initialised) and train.

FAST profile (~35-55 min on an RTX 3060 Ti):
  - HRNet-w32, bottom-up (single animal, no detector), 150 epochs, batch 6
  - backbone initialised from SuperAnimal-Quadruped (with_decoder=False, because
    head_top has no SuperAnimal counterpart -- the 5 keypoint heads train fresh)
  - 90/10 train/test split

Run:  double-click train.bat
Then check evaluation-results/ for train/test pixel error + LabeledImages.
"""
import shutil
from pathlib import Path

import deeplabcut
from deeplabcut.core.engine import Engine
from deeplabcut.modelzoo import build_weight_init

CFG = open(r"E:\TreeShrew_Jumping\analysis\dlc\PROJECT_CONFIG_PATH.txt").read().strip()
PROJ = Path(CFG).parent
MODEL = "hrnet_w32"


def main():
    # clean any half-built dataset/model from a previous run
    for sub in ("training-datasets/iteration-0", "dlc-models-pytorch/iteration-0"):
        p = PROJ / sub
        if p.exists():
            shutil.rmtree(p)

    cfg = deeplabcut.auxiliaryfunctions.read_config(CFG)
    cfg["TrainingFraction"] = [0.9]
    deeplabcut.auxiliaryfunctions.write_config(CFG, cfg)

    wi = build_weight_init(
        cfg=cfg, super_animal="superanimal_quadruped", model_name=MODEL,
        detector_name=None, with_decoder=False, memory_replay=False,
    )

    deeplabcut.create_training_dataset(
        CFG, num_shuffles=1, net_type=MODEL, engine=Engine.PYTORCH,
        weight_init=wi, userfeedback=False,
    )

    deeplabcut.train_network(
        CFG, shuffle=1, device="cuda",
        batch_size=6, epochs=150, save_epochs=50,
    )

    deeplabcut.evaluate_network(CFG, Shuffles=[1], plotting=True)
    print("\ndone -- see evaluation-results/  (train vs test pixel error)")


if __name__ == "__main__":
    main()
