"""Parser de RTTM -> lista de segmentos {start, end, speaker_global}.

RTTM (NIST) por línea:
  SPEAKER <uri> <chan> <start> <dur> <NA> <NA> <speaker> <NA> <NA>
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class RttmSegment:
    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def parse_rttm(path: str | Path) -> List[RttmSegment]:
    """Lee un RTTM y devuelve segmentos ordenados por inicio."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No existe el RTTM: {path}")

    segs: List[RttmSegment] = []
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        parts = line.split()
        if parts[0].upper() != "SPEAKER" or len(parts) < 8:
            continue
        try:
            start = float(parts[3])
            dur = float(parts[4])
        except ValueError as e:  # pragma: no cover - RTTM malformado
            raise ValueError(f"{path}:{ln} start/dur no numéricos: {line}") from e
        speaker = parts[7]
        segs.append(RttmSegment(start=start, end=start + dur, speaker=speaker))

    segs.sort(key=lambda s: s.start)
    return segs


def speakers(segs: List[RttmSegment]) -> List[str]:
    """Etiquetas de hablante únicas, ordenadas."""
    return sorted({s.speaker for s in segs})


def duration_by_speaker(segs: List[RttmSegment]) -> dict:
    """Segundos hablados por cada speaker_global."""
    out: dict = {}
    for s in segs:
        out[s.speaker] = out.get(s.speaker, 0.0) + s.duration
    return out
