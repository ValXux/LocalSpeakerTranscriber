"""Ensambla la salida unificada y el reporte.

Dos modos:
  - build_rows:               RTTM + CSVs externos + offsets (pipeline legacy Google STT)
  - build_rows_from_transcript: RTTM + segmentos Whisper (pipeline self-contained)

Produce: transcript_unificado.{csv,json} + reporte.md
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

from .align import Aligner, UNKNOWN
from .rttm import RttmSegment, duration_by_speaker, speakers

if TYPE_CHECKING:
    from .csv_loader import Segment
    from .transcribe import TranscriptSegment

LOW_CONF = 0.5

# columnas exactas pedidas para el CSV final
CSV_COLUMNS = [
    "speaker_global", "start_global", "end_global", "text",
    "source_csv", "speaker_local_original", "overlap_confidence",
]


@dataclass
class UnifiedRow:
    speaker_global: str
    start_global: float
    end_global: float
    text: str
    source_csv: str
    speaker_local_original: str
    overlap_confidence: float
    # extras (solo en JSON)
    lang: str | None = None
    ambiguous: bool = False
    runner_up: str | None = None

    def csv_dict(self) -> dict:
        d = asdict(self)
        return {k: d[k] for k in CSV_COLUMNS}


def build_rows_from_transcript(
    rttm_segments: List[RttmSegment],
    transcript_segments: "List[TranscriptSegment]",
) -> List[UnifiedRow]:
    """Pipeline self-contained: sin CSVs externos, sin offsets.

    Los timestamps de Whisper ya son globales (audio completo).
    """
    aligner = Aligner(rttm_segments)
    rows: List[UnifiedRow] = []
    for seg in transcript_segments:
        al = aligner.assign(seg.start, seg.end)
        rows.append(UnifiedRow(
            speaker_global=al.speaker_global,
            start_global=round(seg.start, 3),
            end_global=round(seg.end, 3),
            text=seg.text,
            source_csv="",
            speaker_local_original="",
            overlap_confidence=al.overlap_confidence,
            lang=getattr(seg, "lang", None),
            ambiguous=al.ambiguous,
            runner_up=al.runner_up,
        ))
    rows.sort(key=lambda r: r.start_global)
    return rows


def build_rows(
    rttm_segments: List[RttmSegment],
    by_source: "Dict[str, List[Segment]]",
    offsets: Dict[str, float],
) -> List[UnifiedRow]:
    """Pipeline legacy (Google STT CSVs): lleva segmentos a tiempo global."""
    from .csv_loader import Segment  # noqa: F401 — import en runtime para evitar ciclo
    aligner = Aligner(rttm_segments)
    rows: List[UnifiedRow] = []
    for source, segments in by_source.items():
        off = float(offsets.get(source, 0.0))
        for seg in segments:
            gs, ge = seg.start + off, seg.end + off
            al = aligner.assign(gs, ge)
            rows.append(UnifiedRow(
                speaker_global=al.speaker_global,
                start_global=round(gs, 3),
                end_global=round(ge, 3),
                text=seg.text,
                source_csv=f"{source}.csv",
                speaker_local_original=seg.speaker_local,
                overlap_confidence=al.overlap_confidence,
                lang=seg.lang,
                ambiguous=al.ambiguous,
                runner_up=al.runner_up,
            ))
    rows.sort(key=lambda r: r.start_global)
    return rows


def write_outputs(
    rows: List[UnifiedRow],
    rttm_segments: List[RttmSegment],
    out_dir: str | Path,
    *,
    offsets: Dict[str, float] | None = None,
    extra_warnings: List[str] | None = None,
) -> Dict[str, Path]:
    """Escribe csv, json y reporte. Devuelve {nombre: ruta}."""
    import csv as _csv

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": out_dir / "transcript_unificado.csv",
        "json": out_dir / "transcript_unificado.json",
        "report": out_dir / "reporte.md",
    }

    # CSV
    with paths["csv"].open("w", encoding="utf-8", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r.csv_dict())

    # JSON (con extras)
    paths["json"].write_text(
        json.dumps([asdict(r) for r in rows], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Reporte
    paths["report"].write_text(
        _report(rows, rttm_segments, offsets or {}, extra_warnings or []),
        encoding="utf-8",
    )
    return paths


def _report(
    rows: List[UnifiedRow],
    rttm_segments: List[RttmSegment],
    offsets: Dict[str, float],
    extra_warnings: List[str],
) -> str:
    spk = speakers(rttm_segments)
    dur = duration_by_speaker(rttm_segments)
    total_dur = sum(dur.values()) or 1.0

    n = len(rows)
    n_unknown = sum(1 for r in rows if r.speaker_global == UNKNOWN)
    n_low = sum(1 for r in rows if r.overlap_confidence < LOW_CONF)
    n_amb = sum(1 for r in rows if r.ambiguous)

    # texto hablado por speaker_global (según asignación final)
    txt_by_spk: Dict[str, int] = {}
    seg_by_spk: Dict[str, int] = {}
    for r in rows:
        seg_by_spk[r.speaker_global] = seg_by_spk.get(r.speaker_global, 0) + 1
        txt_by_spk[r.speaker_global] = txt_by_spk.get(r.speaker_global, 0) + len(r.text)

    L = []
    L.append("# Reporte de diarización unificada\n")
    L.append(f"- Hablantes detectados (RTTM): **{len(spk)}** -> {', '.join(spk)}")
    L.append(f"- Segmentos de texto unificados: **{n}**")
    pct_unk = 100 * n_unknown / n if n else 0
    pct_low = 100 * n_low / n if n else 0
    pct_amb = 100 * n_amb / n if n else 0
    L.append(f"- UNKNOWN (sin solapamiento): **{n_unknown}** ({pct_unk:.1f}%)")
    L.append(f"- Baja confianza (<{LOW_CONF}): **{n_low}** ({pct_low:.1f}%) — revisar manualmente")
    L.append(f"- Solapamiento de hablantes (ambiguos): **{n_amb}** ({pct_amb:.1f}%)\n")

    L.append("## Duración por hablante (desde el RTTM)\n")
    L.append("| speaker_global | seg hablados | % del habla | nº segmentos texto |")
    L.append("|---|---|---|---|")
    for s in spk:
        d = dur.get(s, 0.0)
        L.append(f"| {s} | {d:.1f}s | {100*d/total_dur:.1f}% | {seg_by_spk.get(s, 0)} |")
    if n_unknown:
        L.append(f"| {UNKNOWN} | — | — | {seg_by_spk.get(UNKNOWN, 0)} |")

    L.append("\n## Offsets aplicados (Caso B)\n")
    if offsets:
        L.append("| source_csv | offset (s) |")
        L.append("|---|---|")
        for k in sorted(offsets):
            L.append(f"| {k}.csv | {offsets[k]:.3f} |")
    else:
        L.append("_(sin offsets: timestamps tratados como globales / Caso A)_")

    L.append("\n## Advertencias\n")
    warns = list(extra_warnings)
    if n_low:
        warns.append(
            f"{n_low} segmentos con solapamiento <{LOW_CONF}: revisar alineación "
            "(posible desfase de offset o segmento a caballo entre turnos)."
        )
    if n_unknown:
        warns.append(
            f"{n_unknown} segmentos sin speaker (silencio/música/VAD): marcados {UNKNOWN}."
        )
    if not warns:
        warns.append("Sin advertencias.")
    for w in warns:
        L.append(f"- {w}")

    return "\n".join(L) + "\n"
