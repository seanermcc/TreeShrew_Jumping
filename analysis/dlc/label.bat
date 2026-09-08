@echo off
setlocal
set "DLC=E:\TreeShrew_Jumping\analysis\dlc"
set "PY=%DLC%\py310\python.exe"
set "HF_HOME=%DLC%\cache\huggingface"
set "TORCH_HOME=%DLC%\cache\torch"
set "DLCLIBRARY=%DLC%\cache\dlclibrary"
cd /d "%DLC%"

echo Checking Python environment...
"%PY%" -c "import deeplabcut, napari; print('  OK - deeplabcut', deeplabcut.__version__, '| napari', napari.__version__)"
if errorlevel 1 (
  echo.
  echo *** The DLC Python environment is not working. Copy everything above and send it back. ***
  echo.
  pause
  exit /b 1
)

echo Launching napari labeling GUI...
"%PY%" label.py
echo.
echo (napari closed)
pause
