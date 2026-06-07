"""Offsets de cada CSV en la línea de tiempo global (Caso B).

Los 8 CSV de Google STT tienen timestamps LOCALES (cada trozo empieza en 0).
Para llevarlos al tiempo global necesitamos el offset = posición de inicio del
trozo dentro del audio completo.

Dos vías (en orden de preferencia):
  1) config/chunk_offsets.json provisto manualmente: {"parte_000.csv": 0.0, ...}
  2) derivar de las duraciones acumuladas de los chunks de audio (ffprobe).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from .audio_prep import probe


def _natural_key(p: Path):
    """Orden natural: parte_2 antes que parte_10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]


def derive_from_chunks(chunks_dir: str | Path, pattern: str = "*.mp3") -> Dict[str, float]:
    """Offsets por duraciones acumuladas de los chunks ordenados naturalmente.

    Devuelve {nombre_chunk_sin_ext: offset_segundos}. La clave es el *stem*
    (p.ej. 'parte_000') para casar luego con el CSV homónimo.
    """
    chunks_dir = Path(chunks_dir)
    files = sorted(chunks_dir.glob(pattern), key=_natural_key)
    if not files:
        raise FileNotFoundError(f"No hay chunks {pattern} en {chunks_dir}")

    offsets: Dict[str, float] = {}
    acc = 0.0
    for f in files:
        offsets[f.stem] = round(acc, 3)
        acc += probe(f).duration
    return offsets


def load_offsets_json(path: str | Path) -> Dict[str, float]:
    """Carga offsets manuales. Las claves se normalizan a *stem* sin extensión."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {Path(k).stem: float(v) for k, v in data.items()}


def save_offsets_json(offsets: Dict[str, float], path: str | Path) -> None:
    """Persiste offsets como {stem.csv: offset} para que sean editables a mano."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serial = {f"{stem}.csv": off for stem, off in offsets.items()}
    path.write_text(json.dumps(serial, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_offsets(
    csv_stems: List[str],
    *,
    offsets_json: Optional[str | Path] = None,
    chunks_dir: Optional[str | Path] = None,
    chunks_pattern: str = "*.mp3",
) -> Dict[str, float]:
    """Resuelve un offset por cada CSV.

    Prioridad: offsets_json -> chunks_dir -> todos 0.0 (Caso A / fallback).
    Cualquier stem sin offset conocido cae a 0.0 con aviso implícito (el caller
    debería loguearlo).
    """
    base: Dict[str, float] = {}
    if offsets_json and Path(offsets_json).is_file():
        base = load_offsets_json(offsets_json)
    elif chunks_dir and Path(chunks_dir).is_dir():
        base = derive_from_chunks(chunks_dir, chunks_pattern)

    return {stem: float(base.get(stem, 0.0)) for stem in csv_stems}
