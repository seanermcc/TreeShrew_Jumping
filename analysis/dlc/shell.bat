@echo off
REM Opens a cmd prompt with the DLC Python on PATH, in the project folder.
REM From here:  python 05_train.py   etc.
set "DLC=E:\TreeShrew_Jumping\analysis\dlc"
cd /d "%DLC%"
set "PATH=%DLC%\py310;%DLC%\py310\Scripts;%PATH%"
set "HF_HOME=%DLC%\cache\huggingface"
set "TORCH_HOME=%DLC%\cache\torch"
set "DLCLIBRARY=%DLC%\cache\dlclibrary"
set "PROMPT=(dlc) $P$G"
echo DLC Python: %DLC%\py310\python.exe
python -c "import deeplabcut; print('deeplabcut', deeplabcut.__version__)"
cmd /k
