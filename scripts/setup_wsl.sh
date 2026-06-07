#!/usr/bin/env bash
# Setup del entorno de transcripcion + diarizacion en WSL2 Ubuntu.
# Requiere sudo en el paso 1. Corre desde /mnt/d/Transcripcion:
#     bash scripts/setup_wsl.sh
set -euo pipefail

cd "$(dirname "$0")/.."   # raiz del proyecto

echo "==> 1/4 dependencias de sistema (sudo)"
sudo apt-get update
sudo apt-get install -y ffmpeg libsndfile1

echo "==> 2/4 uv + venv Python 3.11 (aislado, sin tocar el sistema)"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.11 .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> 3/4 PyTorch CUDA 12.1 (RTX 3050)"
uv pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121

echo "==> 4/4 NeMo + utilidades"
uv pip install -r requirements.txt

echo
echo "OK. Verificacion rapida:"
python - <<'PY'
import torch
print("torch", torch.__version__, "| CUDA disponible:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY
echo
echo "Listo. Activa el entorno con:  source .venv/bin/activate"
echo "Smoke test:"
echo "  python -m src.cli smoke --audio data/audio/mi_audio.mp3 \\"
echo "      --seconds 120 --out output/smoke --model small --language es --batch-size 8"
