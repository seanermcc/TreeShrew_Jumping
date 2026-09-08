@echo off
setlocal
set "DLC=E:\TreeShrew_Jumping\analysis\dlc"
set "PY=%DLC%\py310\python.exe"
cd /d "%DLC%"
"%PY%" -c "import matplotlib; print('matplotlib', matplotlib.__version__); import tkinter; print('tk', tkinter.TkVersion)"
if errorlevel 1 (
  echo *** environment problem - send this back *** & pause & exit /b 1
)
"%PY%" correct_labels.py
echo.
echo (labeler closed - all changes saved)
pause
