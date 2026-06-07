"""Transcripción de audio con faster-whisper (CTranslate2-optimized Whisper).

Import perezoso: faster_whisper solo se carga al llamar transcribe_audio.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    lang: Optional[str] = None


def transcribe_audio(
    audio_wav: str | Path,
    *,
    model: str = "medium",
    language: Optional[str] = None,
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
) -> List[TranscriptSegment]:
    """Transcribe WAV 16k mono. Devuelve segmentos con timestamps globales.

    language=None -> detección automática.
    device=None   -> CUDA si está disponible, sino CPU.
    """
    from faster_whisper import WhisperModel  # type: ignore

    audio_wav = Path(audio_wav)

    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"

    print(f"[transcribe] model={model} device={device} compute_type={compute_type}")

    model_obj = WhisperModel(model, device=device, compute_type=compute_type)

    segments_iter, info = model_obj.transcribe(
        str(audio_wav),
        language=language,
        word_timestamps=False,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=5,
    )

    detected_lang = info.language
    print(
        f"[transcribe] idioma detectado: {detected_lang} "
        f"(prob={info.language_probability:.2f})"
    )

    results: List[TranscriptSegment] = []
    for seg in segments_iter:
        text = seg.text.strip()
        if text:
            results.append(TranscriptSegment(
                start=round(seg.start, 3),
                end=round(seg.end, 3),
                text=text,
                lang=detected_lang,
            ))

    print(f"[transcribe] {len(results)} segmentos")
    return results
