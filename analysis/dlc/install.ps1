$ErrorActionPreference = "Stop"
$uv = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe"
$root = "E:\TreeShrew_Jumping\analysis\dlc"
$venv = "$root\venv"

Write-Output "=== creating venv (py3.10) ==="
& $uv venv --python 3.10 $venv
$py = "$venv\Scripts\python.exe"

Write-Output "=== installing torch (CUDA 12.4) ==="
& $uv pip install --python $py torch torchvision --index-url https://download.pytorch.org/whl/cu124

Write-Output "=== installing deeplabcut[gui] ==="
& $uv pip install --python $py "deeplabcut[gui,modelzoo]"

Write-Output "=== versions ==="
& $py -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
& $py -c "import deeplabcut; print('deeplabcut', deeplabcut.__version__)"
Write-Output "=== DONE ==="
