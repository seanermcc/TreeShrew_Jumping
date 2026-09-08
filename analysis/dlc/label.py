"""Open the napari labelling GUI for the tree-shrew head project.

Run:  venv\\Scripts\\python.exe label.py
"""
import deeplabcut

CFG = open(r"E:\TreeShrew_Jumping\analysis\dlc\PROJECT_CONFIG_PATH.txt").read().strip()

if __name__ == "__main__":
    deeplabcut.label_frames(CFG)
