@echo off
setlocal
set "DLC=E:\TreeShrew_Jumping\analysis\dlc"
set "PY=%DLC%\py310\python.exe"
set "HF_HOME=%DLC%\cache\huggingface"
set "TORCH_HOME=%DLC%\cache\torch"
set "DLCLIBRARY=%DLC%\cache\dlclibrary"
cd /d "%DLC%"
REM Renders labeled overlay videos for all 54 clips (~1 h). Run after analyze.bat.
"%PY%" 06b_labeled_videos.py
echo.
echo === labeled videos are in  TreeShrewHead-sem-2026-08-27\videos\  (*_labeled.mp4) ===
pause
