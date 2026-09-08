@echo off
setlocal
set "AN=E:\TreeShrew_Jumping\analysis"
set "PY=%AN%\dlc\py310\python.exe"
cd /d "%AN%"
"%PY%" -c "import matplotlib; print('matplotlib', matplotlib.__version__); import tkinter; print('tk', tkinter.TkVersion)"
if errorlevel 1 (
  echo *** environment problem - send this back *** & pause & exit /b 1
)
rem  optional: pass jump ids to label just those, e.g.  label_paws.bat v3_03 v3_09
"%PY%" v4_paw_gui.py %*
echo.
echo (labeller closed - saved to v4\paw_contacts_v4.csv)
pause
