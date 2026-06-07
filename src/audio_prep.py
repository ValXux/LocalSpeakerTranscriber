"""Audio probing y conversión a WAV 16 kHz mono PCM-16 (lo que NeMo espera).

Usa ffmpeg/ffprobe del sistema. No depende de torch/NeMo, así que es testeable
de forma aislada.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class AudioError(RuntimeError):
    """Audio corrupto, ausente o con metadatos inesperados."""


@dataclass
class AudioInfo:
    path: str
    duration: float          # segundos
    sample_rate: int
    channels: int
    codec: str
    format_name: str

    def __str__(self) -> str:  # pragma: no cover - solo formato
        return (
            f"{Path(self.path).name}: {self.duration:.2f}s, "
            f"{self.sample_rate} Hz, {self.channels} ch, "
            f"codec={self.codec}, fmt={self.format_name}"
        )


def _require(tool: str) -> str:
    exe = shutil.which(tool)
    if exe is None:
        raise AudioError(
            f"'{tool}' no está en el PATH. Instala ffmpeg "
            "para Windows y confirma con 'where ffmpeg' y 'where ffprobe'."
        )
    return exe


def probe(path: str | Path) -> AudioInfo:
    """Devuelve metadatos del audio vía ffprobe. Aborta si el archivo es inválido."""
    path = Path(path)
    if not path.is_file():
        raise AudioError(f"No existe el archivo de audio: {path}")

    ffprobe = _require("ffprobe")
    cmd = [
        ffprobe, "-v", "error",
        "-show_entries", "format=duration,format_name",
        "-show_entries", "stream=codec_type,codec_name,sample_rate,channels",
        "-of", "json", str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError as e:  # pragma: no cover - I/O
        raise AudioError(f"ffprobe falló sobre {path}: {e.stderr.strip()}") from e

    data = json.loads(out)
    fmt = data.get("format", {})
    astream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "audio"),
        None,
    )
    if astream is None:
        raise AudioError(f"{path} no contiene stream de audio.")

    try:
        duration = float(fmt.get("duration", astream.get("duration", "nan")))
    except (TypeError, ValueError):
        duration = float("nan")
    if not (duration > 0):
        raise AudioError(f"Duración inválida ({duration}) en {path}; ¿audio corrupto?")

    return AudioInfo(
        path=str(path),
        duration=duration,
        sample_rate=int(astream.get("sample_rate", 0) or 0),
        channels=int(astream.get("channels", 0) or 0),
        codec=str(astream.get("codec_name", "?")),
        format_name=str(fmt.get("format_name", "?")),
    )


def to_wav_16k_mono(
    src: str | Path,
    dst: str | Path,
    *,
    sample_rate: int = 16000,
    overwrite: bool = True,
) -> AudioInfo:
    """Convierte cualquier entrada a WAV mono 16 kHz PCM-16. Devuelve info del WAV.

    NeMo (MarbleNet/TitaNet) asume exactamente este formato.
    """
    src, dst = Path(src), Path(dst)
    info_in = probe(src)  # valida la entrada antes de gastar tiempo

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return probe(dst)

    ffmpeg = _require("ffmpeg")
    cmd = [
        ffmpeg, "-y" if overwrite else "-n",
        "-i", str(src),
        "-ac", "1",                 # mono
        "-ar", str(sample_rate),    # 16 kHz
        "-c:a", "pcm_s16le",        # PCM 16-bit
        "-vn",                       # sin video
        str(dst),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:  # pragma: no cover - I/O
        raise AudioError(f"ffmpeg falló convirtiendo {src}: {e.stderr.strip()}") from e

    info_out = probe(dst)
    # sanity: la duración no debería cambiar más de ~0.5s por el reencode
    if abs(info_out.duration - info_in.duration) > 0.5:
        raise AudioError(
            f"Duración cambió tras conversión: {info_in.duration:.2f}s -> "
            f"{info_out.duration:.2f}s ({src})."
        )
    return info_out


def make_clip(src: str | Path, dst: str | Path, seconds: float, start: float = 0.0) -> AudioInfo:
    """Recorta `seconds` desde `start` (para el smoke test). Devuelve info del clip."""
    src, dst = Path(src), Path(dst)
    probe(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _require("ffmpeg")
    cmd = [
        ffmpeg, "-y", "-ss", str(start), "-t", str(seconds),
        "-i", str(src), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-vn",
        str(dst),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return probe(dst)
