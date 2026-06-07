"""Exporta el transcript unificado a un .txt legible tipo diálogo.

speaker_0 -> "Usuario 1", speaker_1 -> "Usuario 2", ... ; UNKNOWN -> "Desconocido".
Junta segmentos consecutivos del mismo hablante en un solo bloque.
Solo usa la stdlib (csv) -> corre en cualquier Python.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional

UNKNOWN_LABEL = "Desconocido"


def _hms(seconds: float) -> str:
    s = int(round(float(seconds)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:01d}:{m:02d}:{sec:02d}"


def _speaker_map(speakers: List[str]) -> Dict[str, str]:
    """speaker_0,speaker_1,... -> Usuario 1, Usuario 2, ... (orden numérico)."""
    def keyf(s: str):
        m = re.search(r"(\d+)$", s)
        return int(m.group(1)) if m else 1_000_000
    ordered = sorted([s for s in speakers if s != "UNKNOWN"], key=keyf)
    mapping = {s: f"Usuario {i+1}" for i, s in enumerate(ordered)}
    mapping["UNKNOWN"] = UNKNOWN_LABEL
    return mapping


def _read_rows(csv_path: str | Path) -> List[dict]:
    with open(csv_path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: float(r["start_global"]))
    return rows


def export_txt(
    csv_path: str | Path,
    out_path: str | Path,
    *,
    title: str = "Transcripción unificada",
    merge_consecutive: bool = True,
    show_timestamps: bool = True,
) -> Path:
    rows = _read_rows(csv_path)
    speakers = sorted({r["speaker_global"] for r in rows})
    smap = _speaker_map(speakers)

    # bloques: junta segmentos consecutivos del mismo speaker_global
    blocks: List[dict] = []
    for r in rows:
        spk = r["speaker_global"]
        text = (r["text"] or "").strip()
        if not text:
            continue
        if (merge_consecutive and blocks and blocks[-1]["spk"] == spk):
            blocks[-1]["text"] += " " + text
            blocks[-1]["end"] = r["end_global"]
        else:
            blocks.append({"spk": spk, "text": text,
                           "start": r["start_global"], "end": r["end_global"]})

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append(f"=== {title} ===")
    leyenda = ", ".join(f"{smap[s]} = {s}" for s in sorted(smap) if s in speakers and s != "UNKNOWN")
    lines.append(f"Hablantes: {leyenda}")
    if "UNKNOWN" in speakers:
        lines.append(f"({UNKNOWN_LABEL} = segmentos sin hablante claro: silencio/solape/VAD)")
    lines.append("")

    for b in blocks:
        who = smap.get(b["spk"], b["spk"])
        if show_timestamps:
            lines.append(f"[{_hms(b['start'])}] {who}: {b['text']}")
        else:
            lines.append(f"{who}: {b['text']}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
