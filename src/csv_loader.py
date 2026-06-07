"""Detección y normalización de los CSV de Google Speech-to-Text.

Google STT exportado a CSV varía:
  * nivel-segmento  : speaker, start, end, transcript      (un turno por fila)
  * nivel-palabra   : word, start_time, end_time, speaker_tag  (una palabra/fila)

Este módulo:
  1) inspecciona los headers reales (insensible a may/min y acentos),
  2) los mapea a un esquema interno {start, end, speaker_local, text, lang},
  3) si vienen a nivel-palabra, agrupa palabras consecutivas del mismo hablante
     en segmentos.

No depende de torch/NeMo: testeable de forma aislada.
"""
from __future__ import annotations

import io
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# --------------------------------------------------------------------------- #
# Esquema interno
# --------------------------------------------------------------------------- #
@dataclass
class Segment:
    start: float            # segundos, tiempo LOCAL del trozo
    end: float
    speaker_local: str      # etiqueta tal cual la dio Google (str para no perder ceros)
    text: str
    lang: Optional[str] = None
    source: Optional[str] = None   # stem del CSV de origen (p.ej. 'parte_000')

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SchemaInfo:
    """Lo que se detectó, para mostrárselo al usuario antes de continuar."""
    source: str
    level: str                      # 'segment' | 'word'
    columns_detected: Dict[str, str]   # campo_interno -> header_original
    n_rows_raw: int
    n_segments: int
    time_format: str                # 'seconds' | 'clock'

    def summary(self) -> str:  # pragma: no cover - formato
        cols = ", ".join(f"{k}<-'{v}'" for k, v in self.columns_detected.items())
        return (
            f"[{self.source}] nivel={self.level} tiempo={self.time_format} "
            f"filas={self.n_rows_raw} -> segmentos={self.n_segments}\n"
            f"          mapeo: {cols}"
        )


# --------------------------------------------------------------------------- #
# Normalización de headers y aliases
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    """minúsculas, sin acentos, sin espacios extra."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.lower().strip().split())


_ALIASES: Dict[str, set] = {
    "start": {
        "start", "start time", "start_time", "starttime", "begin", "inicio",
        "tiempo de inicio", "comienzo", "from",
    },
    "end": {
        "end", "end time", "end_time", "endtime", "fin", "finish", "to",
        "tiempo de finalizacion", "finalizacion", "tiempo final",
    },
    "speaker": {
        "speaker", "speaker tag", "speaker_tag", "speakertag", "spk",
        "speaker label", "speaker_label", "interlocutor",
        "etiqueta de interlocutor", "hablante", "locutor",
    },
    "text": {
        "transcript", "transcription", "transcripcion", "text", "texto",
        "content", "contenido", "sentence", "utterance",
    },
    "word": {"word", "palabra", "token"},
    "lang": {
        "lang", "language", "language code", "language_code", "languagecode",
        "idioma", "codigo de idioma",
    },
    "conf": {"confidence", "confianza", "score"},
    "channel": {"channel", "canal", "channel_tag", "channel tag"},
}


def _build_header_map(columns: List[str]) -> Dict[str, str]:
    """campo_interno -> header_original (primer match gana)."""
    norm_to_orig = {_norm(c): c for c in columns}
    found: Dict[str, str] = {}
    for field, aliases in _ALIASES.items():
        for alias in aliases:
            if alias in norm_to_orig:
                found[field] = norm_to_orig[alias]
                break
    return found


# --------------------------------------------------------------------------- #
# Parseo de tiempos
# --------------------------------------------------------------------------- #
def parse_time(value) -> float:
    """Acepta float-segundos ('28.36', '28.36s') y reloj ('0:00:28.360', '01:28')."""
    if value is None:
        raise ValueError("tiempo vacío")
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().lower().rstrip("s").strip()
    if s == "" or s == "nan":
        raise ValueError("tiempo vacío")
    if ":" in s:
        parts = [float(p) for p in s.split(":")]
        sec = 0.0
        for p in parts:
            sec = sec * 60 + p
        return sec
    return float(s)


def _looks_like_clock(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(20)
    return any(":" in v for v in sample)


# --------------------------------------------------------------------------- #
# Carga / normalización
# --------------------------------------------------------------------------- #
def _read_frame(src) -> pd.DataFrame:
    """Lee desde path o desde un buffer/str de CSV. Maneja texto con comas/comillas."""
    if isinstance(src, (str, Path)) and Path(str(src)).is_file():
        return pd.read_csv(src, dtype=str, keep_default_na=False, skipinitialspace=False)
    if isinstance(src, str):
        return pd.read_csv(io.StringIO(src), dtype=str, keep_default_na=False)
    return pd.read_csv(src, dtype=str, keep_default_na=False)


def normalize_frame(df: pd.DataFrame, source: str = "<mem>") -> Tuple[List[Segment], SchemaInfo]:
    """Convierte un DataFrame crudo en segmentos normalizados + info de esquema."""
    hmap = _build_header_map(list(df.columns))

    # campos mínimos
    if "start" not in hmap or "end" not in hmap:
        raise ValueError(
            f"[{source}] no encuentro columnas de inicio/fin. Headers: {list(df.columns)}"
        )
    if "speaker" not in hmap:
        raise ValueError(
            f"[{source}] no encuentro columna de hablante. Headers: {list(df.columns)}"
        )

    is_word_level = "word" in hmap and "text" not in hmap
    text_col = hmap.get("text") or hmap.get("word")
    if text_col is None:
        raise ValueError(f"[{source}] no encuentro columna de texto ni de palabra.")

    time_format = "clock" if _looks_like_clock(df[hmap["start"]]) else "seconds"

    rows = []
    for _, r in df.iterrows():
        raw_s, raw_e = r[hmap["start"]], r[hmap["end"]]
        try:
            s, e = parse_time(raw_s), parse_time(raw_e)
        except ValueError:
            continue  # fila sin tiempos válidos -> se descarta
        rows.append({
            "start": s,
            "end": e,
            "speaker_local": str(r[hmap["speaker"]]).strip(),
            "text": str(r[text_col]).strip(),
            "lang": str(r[hmap["lang"]]).strip() if "lang" in hmap else None,
        })

    if is_word_level:
        segments = _group_words(rows, source)
    else:
        segments = [
            Segment(source=source, **row) for row in rows if row["text"] != ""
        ]

    info = SchemaInfo(
        source=source,
        level="word" if is_word_level else "segment",
        columns_detected={k: v for k, v in hmap.items()},
        n_rows_raw=len(df),
        n_segments=len(segments),
        time_format=time_format,
    )
    return segments, info


def _group_words(rows: List[dict], source: str) -> List[Segment]:
    """Agrupa palabras consecutivas con el mismo speaker_local en un segmento."""
    segments: List[Segment] = []
    cur: Optional[dict] = None
    for w in rows:
        if w["text"] == "":
            continue
        if cur is None or w["speaker_local"] != cur["speaker_local"]:
            if cur is not None:
                segments.append(_finish(cur, source))
            cur = {
                "speaker_local": w["speaker_local"],
                "start": w["start"],
                "end": w["end"],
                "lang": w["lang"],
                "words": [w["text"]],
            }
        else:
            cur["end"] = w["end"]
            cur["words"].append(w["text"])
    if cur is not None:
        segments.append(_finish(cur, source))
    return segments


def _finish(cur: dict, source: str) -> Segment:
    return Segment(
        start=cur["start"], end=cur["end"], speaker_local=cur["speaker_local"],
        text=" ".join(cur["words"]).strip(), lang=cur["lang"], source=source,
    )


def load_csv(path: str | Path) -> Tuple[List[Segment], SchemaInfo]:
    """Carga y normaliza un CSV. `source` = stem del archivo (p.ej. 'parte_000')."""
    path = Path(path)
    df = _read_frame(path)
    segments, info = normalize_frame(df, source=path.stem)
    return segments, info


def load_dir(transcripts_dir: str | Path) -> Tuple[Dict[str, List[Segment]], List[SchemaInfo]]:
    """Carga todos los .csv de un directorio. Devuelve {stem: segmentos}, [esquemas]."""
    transcripts_dir = Path(transcripts_dir)
    files = sorted(transcripts_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No hay CSV en {transcripts_dir}")
    by_source: Dict[str, List[Segment]] = {}
    schemas: List[SchemaInfo] = []
    for f in files:
        segs, info = load_csv(f)
        by_source[f.stem] = segs
        schemas.append(info)
    return by_source, schemas
