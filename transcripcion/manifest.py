"""Genera el manifest JSONL que el diarizador de NeMo necesita.

NeMo lee una línea JSON por audio. Para inferencia de diarización:
  audio_filepath, offset, duration, label='infer', text='-',
  num_speakers (o null), rttm_filepath/uem_filepath (null en inferencia pura).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def write_manifest(
    audio_filepath: str | Path,
    out_path: str | Path,
    *,
    duration: Optional[float] = None,
    num_speakers: Optional[int] = None,
    offset: float = 0.0,
) -> Path:
    """Escribe un manifest de una sola entrada y devuelve su ruta.

    `audio_filepath` debe ser accesible por NeMo desde el sistema que ejecuta
    Python. En Windows nativo se usa la ruta local tal como la resuelve `Path`.
    """
    audio_filepath = Path(audio_filepath)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "audio_filepath": str(audio_filepath),
        "offset": float(offset),
        "duration": float(duration) if duration is not None else None,
        "label": "infer",
        "text": "-",
        "num_speakers": int(num_speakers) if num_speakers is not None else None,
        "rttm_filepath": None,
        "uem_filepath": None,
    }
    out_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    return out_path
