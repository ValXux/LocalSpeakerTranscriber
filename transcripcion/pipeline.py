"""Pipeline self-contained: audio -> WAV -> diarize + transcribe -> salida unificada.

Reemplaza el flujo antiguo que requería CSVs de Google STT.
Puede ser llamado desde CLI o desde la app Gradio.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import audio_prep
from .consolidate import build_rows_from_transcript, write_outputs
from .export_txt import export_txt
from .rttm import parse_rttm

DEFAULT_CONFIG = Path(__file__).parent.parent / "config" / "diarizer.yaml"


def run_full_pipeline(
    audio_path: str | Path,
    out_dir: str | Path,
    *,
    whisper_model: str = "medium",
    language: Optional[str] = None,
    num_speakers: Optional[int] = None,
    max_num_speakers: int = 20,
    device: Optional[str] = None,
    batch_size: int = 16,
    config_path: str | Path | None = None,
    title: str = "Transcripción",
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> Dict:
    """Corre el pipeline completo y devuelve rutas de salida + filas generadas.

    Retorna:
        {
          "paths": {"csv": Path, "json": Path, "report": Path, "txt": Path, "wav": Path},
          "rows": List[UnifiedRow],
          "rttm_segments": List[RttmSegment],
          "num_speakers": int,
        }
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG

    def _progress(pct: float, msg: str) -> None:
        print(f"[pipeline] {pct:.0%} {msg}")
        if progress_cb:
            progress_cb(pct, msg)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. prep audio
    _progress(0.05, "Preparando audio (16k mono)...")
    info = audio_prep.probe(audio_path)
    _progress(0.08, f"Audio: {info.duration:.1f}s, {info.sample_rate}Hz, {info.channels}ch")

    if info.sample_rate != 16000 or info.channels != 1:
        wav = out_dir / "audio_16k.wav"
        audio_prep.to_wav_16k_mono(audio_path, wav)
    else:
        wav = Path(audio_path)
    _progress(0.12, "Audio listo.")

    # 2. diarize
    _progress(0.14, "Iniciando diarización NeMo (VAD + TitaNet + clustering)...")
    from .diarize import diarize_audio
    rttm_path = diarize_audio(
        wav, out_dir,
        config_path=str(config_path),
        num_speakers=num_speakers,
        max_num_speakers=max_num_speakers,
        device=device,
        batch_size=batch_size,
    )
    rttm_segments = parse_rttm(rttm_path)
    num_spk = len({s.speaker for s in rttm_segments})
    _progress(0.55, f"Diarización completa: {num_spk} hablantes detectados.")

    # 3. transcribe
    _progress(0.57, f"Transcribiendo con Whisper ({whisper_model})...")
    from .transcribe import transcribe_audio
    transcript_segments = transcribe_audio(
        wav,
        model=whisper_model,
        language=language,
        device=device,
    )
    _progress(0.85, f"Transcripción completa: {len(transcript_segments)} segmentos.")

    # 4. align + consolidate
    _progress(0.87, "Alineando hablantes con texto...")
    rows = build_rows_from_transcript(rttm_segments, transcript_segments)

    paths = write_outputs(rows, rttm_segments, out_dir)

    # 5. export txt legible
    txt_path = out_dir / "transcripcion.txt"
    export_txt(paths["csv"], txt_path, title=title)
    paths["txt"] = txt_path
    paths["wav"] = wav

    _progress(1.0, "Pipeline completo.")
    return {
        "paths": paths,
        "rows": rows,
        "rttm_segments": rttm_segments,
        "num_speakers": num_spk,
    }
