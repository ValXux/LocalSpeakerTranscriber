# Setup del entorno de transcripcion + diarizacion en Windows nativo.
# Ejecutar desde PowerShell, en la raiz del proyecto:
#   powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1

[CmdletBinding()]
param(
    [ValidateSet("cuda", "cpu")]
    [string]$Torch = "cuda",

    [string]$PythonVersion = "3.11",

    [string]$Venv = ".venv-win"
)

$ErrorActionPreference = "Stop"

function Require-Command {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [string]$InstallHint
    )

    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -eq $cmd) {
        throw "'$Name' no esta en PATH. $InstallHint"
    }
    return $cmd.Source
}

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

Write-Host "==> 1/5 verificando herramientas de Windows"
$uv = Require-Command "uv" "Instala uv desde https://docs.astral.sh/uv/getting-started/installation/."
$ffmpeg = Require-Command "ffmpeg" "Instala ffmpeg y confirma con: where ffmpeg"
$ffprobe = Require-Command "ffprobe" "Instala ffmpeg y confirma con: where ffprobe"
Write-Host "uv:      $uv"
Write-Host "ffmpeg:  $ffmpeg"
Write-Host "ffprobe: $ffprobe"

$nvidia = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
if ($Torch -eq "cuda" -and $null -eq $nvidia) {
    Write-Warning "Se pidio CUDA, pero nvidia-smi no esta en PATH. Se instalara PyTorch CUDA igual; la verificacion final dira si hay GPU."
}
elseif ($null -ne $nvidia) {
    Write-Host "nvidia-smi: $($nvidia.Source)"
}

Write-Host "==> 2/5 instalando Python $PythonVersion con uv si falta"
uv python install $PythonVersion

Write-Host "==> 3/5 creando entorno Windows $Venv"
$PythonExe = Join-Path $ProjectRoot "$Venv\Scripts\python.exe"
if (Test-Path $PythonExe) {
    Write-Host "Reusando entorno existente: $PythonExe"
}
else {
    uv venv --python $PythonVersion $Venv
}

if (-not (Test-Path $PythonExe)) {
    throw "No se encontro $PythonExe. El entorno debe ser Windows y contener Scripts\python.exe."
}

Write-Host "==> 4/5 instalando PyTorch ($Torch)"
if ($Torch -eq "cuda") {
    uv pip install --python $PythonExe torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
}
else {
    uv pip install --python $PythonExe torch==2.3.1 torchaudio==2.3.1
}

Write-Host "==> 5/5 instalando dependencias del proyecto"
uv pip install --python $PythonExe -r requirements.txt

Write-Host ""
Write-Host "==> Verificacion rapida"
& $PythonExe --version
& $PythonExe -c "import torch, torchaudio; print('torch', torch.__version__, '| torchaudio', torchaudio.__version__, '| CUDA disponible:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
& $PythonExe -c "from transcripcion.diarize import _load_clusterdiarizer; _load_clusterdiarizer(); print('NeMo diarizer import OK')"
& $PythonExe -m transcripcion.cli --help | Out-Null
Write-Host "CLI OK"

Write-Host ""
Write-Host "Listo. Activa el entorno con:"
Write-Host "  .\$Venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Smoke test sugerido:"
Write-Host "  python -m transcripcion.cli smoke --audio data/audio/mi_audio.mp3 --seconds 120 --out output/smoke --model small --language es --batch-size 4"
