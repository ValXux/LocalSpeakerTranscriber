"""Pipeline self-contained de transcripción con detección de hablantes.

Pipeline nuevo (sin dependencias externas):
  audio_prep -> diarize (NeMo) -> transcribe (Whisper) -> align -> consolidate

Pipeline legacy (conservado para CSVs de Google STT):
  csv_loader -> (offsets) -> align -> consolidate
"""

from __future__ import annotations

import os

if os.name == "nt":
    # Windows puede cargar OpenMP desde varias ruedas nativas (torch/ctranslate2).
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.2.0"
