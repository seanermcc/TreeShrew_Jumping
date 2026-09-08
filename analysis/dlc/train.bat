@echo off
setlocal
set "DLC=E:\TreeShrew_Jumping\analysis\dlc"
set "PY=%DLC%\py310\python.exe"
set "HF_HOME=%DLC%\cache\huggingface"
set "TORCH_HOME=%DLC%\cache\torch"
set "DLCLIBRARY=%DLC%\cache\dlclibrary"
cd /d "%DLC%"
REM Build the training dataset (SuperAnimal-initialized) and train. GPU, ~1-3 h.
REM Run this only AFTER you have finished correcting labels in napari.
"%PY%" 05_train.py
echo.
echo === training finished - press any key to close ===
pause >nul
