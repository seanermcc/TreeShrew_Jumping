"""
Step 5a: drop training frames the labeller left completely empty.

During correction you removed all 5 points from ~20 frames where the animal
is an unlabelable blur / out of view. Those rows carry no supervision signal
and can trip up create_training_dataset, so remove them from each clip's
CollectedData before building the training set. (The .png files are left in
place but unreferenced.)
"""
import glob
import os

import numpy as np
import pandas as pd

CFG = open(r"E:\TreeShrew_Jumping\analysis\dlc\PROJECT_CONFIG_PATH.txt").read().strip()
LD = os.path.join(os.path.dirname(CFG), "labeled-data")
SCORER = "sem"


def main():
    total_dropped = 0
    for h5 in sorted(glob.glob(os.path.join(LD, "*", f"CollectedData_{SCORER}.h5"))):
        df = pd.read_hdf(h5)
        empty = df.isna().all(axis=1)
        n = int(empty.sum())
        if n:
            df = df[~empty]
            df.to_hdf(h5, key="df_with_missing", mode="w")
            df.to_csv(h5[:-3] + ".csv")
            total_dropped += n
            print(f"{os.path.basename(os.path.dirname(h5)):16s} dropped {n} empty "
                  f"frame(s) -> {len(df)} left")
    print(f"\ndropped {total_dropped} empty frames total")


if __name__ == "__main__":
    main()
