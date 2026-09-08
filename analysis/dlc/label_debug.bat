@echo off
setlocal enabledelayedexpansion
set "DLC=E:\TreeShrew_Jumping\analysis\dlc"
set "PY=%DLC%\py310\python.exe"
set "HF_HOME=%DLC%\cache\huggingface"
set "TORCH_HOME=%DLC%\cache\torch"
set "DLCLIBRARY=%DLC%\cache\dlclibrary"
set "LOG=%DLC%\label_debug.log"
cd /d "%DLC%"

echo ==== %date% %time% ==== > "%LOG%"

echo. >> "%LOG%"
echo ==== [1] napari sys_info ==== >> "%LOG%"
"%PY%" -X faulthandler -c "import napari.utils.info as i; print(i.sys_info())" >> "%LOG%" 2>&1

echo. >> "%LOG%"
echo ==== [2] launch napari, normal GL ==== >> "%LOG%"
"%PY%" -X faulthandler label.py >> "%LOG%" 2>&1
echo ---- exit code: !errorlevel! ---- >> "%LOG%"

echo. >> "%LOG%"
echo ==== [3] launch napari, software GL (QT_OPENGL=software) ==== >> "%LOG%"
set "QT_OPENGL=software"
set "QT_QUICK_BACKEND=software"
"%PY%" -X faulthandler label.py >> "%LOG%" 2>&1
echo ---- exit code: !errorlevel! ---- >> "%LOG%"
set "QT_OPENGL="
set "QT_QUICK_BACKEND="

echo. >> "%LOG%"
echo ==== [4] Qt platform plugin probe ==== >> "%LOG%"
set "QT_DEBUG_PLUGINS=1"
"%PY%" -c "from qtpy.QtWidgets import QApplication; a=QApplication([]); print('Qt app OK')" >> "%LOG%" 2>&1
set "QT_DEBUG_PLUGINS="

echo.
echo Opening the log - please send its contents back.
start notepad "%LOG%"
pause
