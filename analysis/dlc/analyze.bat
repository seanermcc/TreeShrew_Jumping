@echo off
setlocal
set "DLC=E:\TreeShrew_Jumping\analysis\dlc"
set "PY=%DLC%\py310\python.exe"
set "HF_HOME=%DLC%\cache\huggingface"
set "TORCH_HOME=%DLC%\cache\torch"
set "DLCLIBRARY=%DLC%\cache\dlclibrary"
set "LOG=%DLC%\analyze.log"
cd /d "%DLC%"
REM Runs the trained model on all 54 clips, then exports head position + angle.
REM analyze_videos skips clips already done, so it is safe to re-run if interrupted.

echo Logging to %LOG%
"%PY%" 06_analyze.py > "%LOG%" 2>&1
if errorlevel 1 (
  echo.
  echo *** 06_analyze.py FAILED - opening the log, send me the end of it ***
  start notepad "%LOG%"
  pause
  exit /b 1
)
"%PY%" 07_head_pose.py >> "%LOG%" 2>&1
if errorlevel 1 (
  echo *** 07_head_pose.py FAILED - see %LOG% *** & start notepad "%LOG%" & pause & exit /b 1
)
echo.
echo === analysis done -- head_pose\ and head3d\ have the CSVs ===
echo === visual QC videos:  label_videos.bat  (all 54, slow)  or
echo ===   py310\python.exe 06b_labeled_videos.py cam2_jump_23 cam3_jump_23 ...  (pick clips)
pause
