@echo off
REM Render labeled QC videos for a chosen set of jumps.
REM Edit the JUMPS line below (space-separated jump numbers), then double-click.
setlocal enabledelayedexpansion
set "DLC=E:\TreeShrew_Jumping\analysis\dlc"
set "PY=%DLC%\py310\python.exe"
set "HF_HOME=%DLC%\cache\huggingface"
set "TORCH_HOME=%DLC%\cache\torch"
cd /d "%DLC%"

set "JUMPS=2 10 23"

set "CLIPS="
for %%J in (%JUMPS%) do (
  set "N=0%%J"
  set "N=!N:~-2!"
  set "CLIPS=!CLIPS! cam2_jump_!N! cam3_jump_!N!"
)
echo Rendering QC videos for jumps: %JUMPS%
"%PY%" 06b_labeled_videos.py %CLIPS%
echo.
echo === done -- TreeShrewHead-sem-2026-08-27\videos\*_p60_labeled.mp4 ===
pause
