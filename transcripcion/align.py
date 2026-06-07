"""Alineación por máximo solapamiento temporal.

Para cada segmento de texto (ya en tiempo GLOBAL) se busca el/los segmento(s)
RTTM que más solapan y se asigna ese speaker_global. La fracción de solapamiento
(overlap / duración_del_segmento) es la confianza.

Casos borde:
  * sin solapamiento (silencio/música) -> speaker_global = UNKNOWN, conf = 0.
  * varios hablantes solapan -> se asigna el de mayor solapamiento y se marca
    `ambiguous=True` con el runner-up.
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import List, Optional

from .rttm import RttmSegment

UNKNOWN = "UNKNOWN"


@dataclass
class Alignment:
    speaker_global: str
    overlap_confidence: float       # 0..1  (overlap / dur segmento)
    overlap_seconds: float
    ambiguous: bool = False
    runner_up: Optional[str] = None
    runner_up_seconds: float = 0.0


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


class Aligner:
    """Indexa el RTTM una vez para alinear muchos segmentos eficientemente."""

    def __init__(self, rttm_segments: List[RttmSegment]):
        self.segs = sorted(rttm_segments, key=lambda s: s.start)
        self._starts = [s.start for s in self.segs]
        # fin máximo acumulado para poder cortar la búsqueda hacia atrás
        self._max_end = 0.0
        for s in self.segs:
            self._max_end = max(self._max_end, s.end)

    def assign(self, start: float, end: float) -> Alignment:
        dur = max(end - start, 1e-9)
        if not self.segs:
            return Alignment(UNKNOWN, 0.0, 0.0)

        # candidatos: RTTM cuyo inicio < end. Se barre hacia atrás hasta que
        # ya no puedan solapar (start_rttm + nada alcanza). Como las duraciones
        # varían, se barre desde el primero hasta el último con start < end.
        hi = bisect_left(self._starts, end)
        per_speaker: dict = {}
        # límite inferior: ningún segmento que termine antes de `start` solapa.
        # se recorre [0, hi) y se filtra por end > start.
        for i in range(hi):
            seg = self.segs[i]
            if seg.end <= start:
                continue
            ov = _overlap(start, end, seg.start, seg.end)
            if ov > 0:
                per_speaker[seg.speaker] = per_speaker.get(seg.speaker, 0.0) + ov

        if not per_speaker:
            return Alignment(UNKNOWN, 0.0, 0.0)

        ranked = sorted(per_speaker.items(), key=lambda kv: kv[1], reverse=True)
        best_spk, best_ov = ranked[0]
        runner, runner_ov = (ranked[1] if len(ranked) > 1 else (None, 0.0))
        conf = min(best_ov / dur, 1.0)
        return Alignment(
            speaker_global=best_spk,
            overlap_confidence=round(conf, 4),
            overlap_seconds=round(best_ov, 3),
            ambiguous=len(ranked) > 1,
            runner_up=runner,
            runner_up_seconds=round(runner_ov, 3),
        )


def assign_speaker(start: float, end: float, rttm_segments: List[RttmSegment]) -> Alignment:
    """Conveniencia para un solo segmento (los tests la usan)."""
    return Aligner(rttm_segments).assign(start, end)
