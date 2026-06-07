"""Diarización del audio completo con el cascaded clustering diarizer de NeMo.

VAD (MarbleNet) -> embeddings (TitaNet-large) -> clustering espectral -> RTTM.
Una sola pasada sobre las ~2h => speakers globales consistentes.

Este es el único módulo que importa torch/NeMo. El import es perezoso para que
el resto del pipeline (csv_loader, align, etc.) funcione sin el stack pesado.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

from .audio_prep import probe
from .manifest import write_manifest


def _preload_windows_nemo_dependencies() -> None:
    if os.name != "nt":
        return

    # NeMo importa transformers/datasets en cascada; en Windows, cargar pyarrow
    # serialmente antes de torch evita access violations por imports nativos dobles.
    import pyarrow  # noqa: F401
    import pyarrow.dataset  # noqa: F401
    import datasets  # noqa: F401
    import transformers  # noqa: F401


def _select_device(requested: Optional[str]) -> str:
    import torch  # import perezoso
    if requested in ("cuda", "cpu"):
        if requested == "cuda" and not torch.cuda.is_available():
            print("[diarize] AVISO: se pidió CUDA pero no hay GPU; usando CPU.")
            return "cpu"
        return requested
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"[diarize] GPU detectada: {name}")
        return "cuda"
    print(
        "[diarize] AVISO: sin GPU NVIDIA. Corriendo en CPU. "
        "Diarizar ~2h en CPU puede tardar HORAS."
    )
    return "cpu"


def _load_clusterdiarizer():
    _preload_windows_nemo_dependencies()

    # NeMo 2.x: la clase es `ClusteringDiarizer` (no `ClusterDiarizer`).
    try:
        from nemo.collections.asr.models import ClusteringDiarizer  # type: ignore
    except ImportError:
        from nemo.collections.asr.models.clustering_diarizer import (  # type: ignore
            ClusteringDiarizer,
        )
    return ClusteringDiarizer


def diarize_audio(
    audio_wav: str | Path,
    out_dir: str | Path,
    *,
    config_path: str | Path,
    num_speakers: Optional[int] = None,
    max_num_speakers: int = 20,
    device: Optional[str] = None,
    batch_size: int = 16,
) -> Path:
    """Diariza `audio_wav` (WAV 16k mono) y devuelve la ruta al RTTM resultante.

    Si `num_speakers` se pasa, activa modo oracle (clustering forzado a N).
    """
    audio_wav = Path(audio_wav)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    info = probe(audio_wav)
    if info.sample_rate != 16000 or info.channels != 1:
        raise ValueError(
            f"{audio_wav} debe ser WAV 16k mono; es {info.sample_rate}Hz "
            f"{info.channels}ch. Pasa primero por audio_prep.to_wav_16k_mono."
        )

    _preload_windows_nemo_dependencies()
    dev = _select_device(device)

    manifest_path = out_dir / "manifest.json"
    write_manifest(
        audio_wav, manifest_path,
        duration=info.duration, num_speakers=num_speakers,
    )

    cfg = OmegaConf.load(str(config_path))
    cfg.device = dev
    cfg.batch_size = int(batch_size)
    if os.name == "nt":
        cfg.num_workers = 0
    cfg.diarizer.manifest_filepath = str(manifest_path)
    cfg.diarizer.out_dir = str(out_dir)
    cfg.diarizer.clustering.parameters.max_num_speakers = int(max_num_speakers)
    cfg.diarizer.clustering.parameters.oracle_num_speakers = num_speakers is not None

    print(
        f"[diarize] device={dev} oracle={num_speakers is not None} "
        f"max_spk={max_num_speakers} dur={info.duration:.1f}s"
    )

    ClusterDiarizer = _load_clusterdiarizer()
    diarizer = ClusterDiarizer(cfg=cfg)
    diarizer.diarize()

    produced = out_dir / "pred_rttms" / f"{audio_wav.stem}.rttm"
    if not produced.is_file():
        # NeMo a veces usa el uniq_id del manifest; buscamos cualquier .rttm
        candidates = list((out_dir / "pred_rttms").glob("*.rttm"))
        if not candidates:
            raise FileNotFoundError(
                f"NeMo no produjo RTTM en {out_dir/'pred_rttms'}."
            )
        produced = candidates[0]

    final = out_dir / "diarization.rttm"
    shutil.copyfile(produced, final)
    print(f"[diarize] RTTM -> {final}")
    return final
