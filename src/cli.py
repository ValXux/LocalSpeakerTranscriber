"""Orquestador del pipeline.

Pipeline self-contained (nuevo):
  run         audio -> WAV -> diarize (NeMo) -> transcribe (Whisper) -> salida
  transcribe  WAV -> transcript con segmentos y timestamps (solo Whisper)

Pipeline legacy (conservado para compatibilidad con CSVs de Google STT):
  align       RTTM + CSVs -> salida unificada

Utilidades:
  inspect     probe audio
  prep        audio -> WAV 16k mono
  diarize     WAV -> RTTM (NeMo)
  export-txt  transcript_unificado.csv -> .txt tipo diálogo
  smoke       recorta el audio y corre el pipeline sobre el clip

Ej (nuevo):
  python -m src.cli run --audio reunion.mp3 --out output/
  python -m src.cli run --audio reunion.mp3 --out output/ --model large-v3 --num-speakers 3

Ej (legacy):
  python -m src.cli align --rttm output/diarization.rttm \
      --transcripts partido/transcripts/csv --out output/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from . import audio_prep

DEFAULT_CONFIG = "config/diarizer.yaml"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _print_schemas(schemas):
    print("\n=== Esquema detectado por CSV ===")
    for s in schemas:
        print(s.summary())


def _resolve_offsets(csv_stems, args, persist_to: Optional[str] = None):
    from . import offsets as offmod
    oj = getattr(args, "offsets", None)
    chunks = getattr(args, "chunks", None)

    if oj and Path(oj).is_file():
        base = offmod.load_offsets_json(oj)
        off = {s: float(base.get(s, 0.0)) for s in csv_stems}
        print(f"[offsets] usando {oj} (no se regenera)")
    elif chunks and Path(chunks).is_dir():
        base = offmod.derive_from_chunks(chunks, getattr(args, "chunks_pattern", "*.mp3"))
        off = {s: float(base.get(s, 0.0)) for s in csv_stems}
        if persist_to:
            offmod.save_offsets_json(off, persist_to)
            print(f"[offsets] derivados de chunks -> {persist_to}")
    else:
        off = {s: 0.0 for s in csv_stems}
        print("[offsets] sin fuente: todos 0.0 (Caso A)")

    nonzero = sum(1 for v in off.values() if v != 0.0)
    print(f"[offsets] {len(off)} CSVs, {nonzero} con offset != 0")
    return off


def _align_pipeline(rttm_path, transcripts_dir, out_dir, offsets, extra_warnings=None):
    from .rttm import parse_rttm
    from . import csv_loader
    from .consolidate import build_rows, write_outputs
    rttm_segments = parse_rttm(rttm_path)
    by_source, schemas = csv_loader.load_dir(transcripts_dir)
    _print_schemas(schemas)
    rows = build_rows(rttm_segments, by_source, offsets)
    paths = write_outputs(
        rows, rttm_segments, out_dir, offsets=offsets, extra_warnings=extra_warnings,
    )
    print(f"\n[ok] {len(rows)} segmentos unificados.")
    for k, p in paths.items():
        print(f"  {k}: {p}")
    return paths


# --------------------------------------------------------------------------- #
# subcomandos
# --------------------------------------------------------------------------- #
def cmd_inspect(args):
    info = audio_prep.probe(args.audio)
    print("=== Audio ===")
    print(info)
    if getattr(args, "transcripts", None):
        from . import csv_loader
        _, schemas = csv_loader.load_dir(args.transcripts)
        _print_schemas(schemas)
        by_source, _ = csv_loader.load_dir(args.transcripts)
        print("\n=== Rango temporal LOCAL por CSV ===")
        for src, segs in by_source.items():
            if segs:
                print(f"  {src}: {segs[0].start:.2f}s .. {segs[-1].end:.2f}s ({len(segs)} seg)")


def cmd_offsets(args):
    from . import offsets as offmod
    off = offmod.derive_from_chunks(args.chunks, args.chunks_pattern)
    offmod.save_offsets_json(off, args.out)
    print(f"Offsets derivados -> {args.out}")
    for stem, v in off.items():
        print(f"  {stem}.csv -> {v:.3f}s")


def cmd_prep(args):
    info = audio_prep.to_wav_16k_mono(args.audio, args.out_wav)
    print(f"WAV listo: {info}")


def cmd_diarize(args):
    from .diarize import diarize_audio
    wav = Path(args.audio)
    info = audio_prep.probe(wav)
    if info.sample_rate != 16000 or info.channels != 1:
        wav16 = Path(args.out) / "audio_16k.wav"
        print("[prep] convirtiendo a WAV 16k mono...")
        audio_prep.to_wav_16k_mono(wav, wav16)
        wav = wav16
    rttm = diarize_audio(
        wav, args.out, config_path=args.config,
        num_speakers=args.num_speakers, max_num_speakers=args.max_num_speakers,
        device=args.device, batch_size=args.batch_size,
    )
    print(f"RTTM: {rttm}")


def cmd_transcribe(args):
    """Solo Whisper: WAV -> JSON con segmentos y timestamps."""
    import json
    from .transcribe import transcribe_audio
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lang = None if args.language == "auto" else args.language
    segments = transcribe_audio(
        args.audio,
        model=args.model,
        language=lang,
        device=args.device,
    )

    out_json = out / "transcript_whisper.json"
    out_json.write_text(
        json.dumps(
            [{"start": s.start, "end": s.end, "text": s.text, "lang": s.lang} for s in segments],
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[transcribe] {len(segments)} segmentos -> {out_json}")


def cmd_run(args):
    """Pipeline self-contained: audio -> diarize + transcribe -> salida."""
    from .pipeline import run_full_pipeline
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lang = None if getattr(args, "language", None) in (None, "auto") else args.language

    result = run_full_pipeline(
        args.audio,
        out,
        whisper_model=getattr(args, "model", "medium"),
        language=lang,
        num_speakers=args.num_speakers,
        max_num_speakers=args.max_num_speakers,
        device=args.device,
        batch_size=args.batch_size,
        config_path=args.config,
        title=Path(args.audio).stem,
    )

    paths = result["paths"]
    print(f"\n[ok] {len(result['rows'])} segmentos | {result['num_speakers']} hablantes")
    for k, p in paths.items():
        if isinstance(p, Path):
            print(f"  {k}: {p}")


def cmd_align(args):
    """Pipeline legacy: RTTM + CSVs -> salida unificada."""
    from . import csv_loader
    by_source, _ = csv_loader.load_dir(args.transcripts)
    off = _resolve_offsets(list(by_source.keys()), args, persist_to=args.save_offsets)
    _align_pipeline(args.rttm, args.transcripts, args.out, off)


def cmd_export_txt(args):
    from .export_txt import export_txt
    p = export_txt(
        args.input, args.out,
        title=args.title,
        merge_consecutive=not args.no_merge,
        show_timestamps=not args.no_timestamps,
    )
    print(f"TXT -> {p}")


def cmd_smoke(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clip = out / "smoke_clip.wav"
    info = audio_prep.make_clip(args.audio, clip, seconds=args.seconds, start=args.start)
    print(f"[smoke] clip {info.duration:.1f}s -> {clip}")
    args.audio = str(clip)
    cmd_run(args)


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="src.cli",
        description="Pipeline self-contained de transcripción con detección de hablantes",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_diar_opts(sp):
        sp.add_argument("--config", default=DEFAULT_CONFIG)
        sp.add_argument("--num-speakers", type=int, default=None, dest="num_speakers",
                        help="modo oracle: fuerza N hablantes")
        sp.add_argument("--max-num-speakers", type=int, default=20, dest="max_num_speakers")
        sp.add_argument("--device", choices=["cuda", "cpu"], default=None)
        sp.add_argument("--batch-size", type=int, default=16, dest="batch_size")

    def add_whisper_opts(sp):
        sp.add_argument("--model", default="medium",
                        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                        help="modelo Whisper (default: medium)")
        sp.add_argument("--language", default=None,
                        help="código de idioma (es, en, pt...) o None para auto-detect")

    def add_offset_opts(sp):
        sp.add_argument("--chunks", default=None)
        sp.add_argument("--chunks-pattern", default="*.mp3", dest="chunks_pattern")
        sp.add_argument("--offsets", default="config/chunk_offsets.json")
        sp.add_argument("--save-offsets", default="config/chunk_offsets.json",
                        dest="save_offsets")

    # ── run (nuevo, self-contained) ────────────────────────────────────────── #
    sp = sub.add_parser("run", help="pipeline completo: audio -> transcripción con hablantes")
    sp.add_argument("--audio", required=True)
    sp.add_argument("--out", required=True)
    add_diar_opts(sp)
    add_whisper_opts(sp)
    sp.set_defaults(func=cmd_run)

    # ── transcribe (solo Whisper) ──────────────────────────────────────────── #
    sp = sub.add_parser("transcribe", help="solo Whisper: WAV -> JSON con segmentos")
    sp.add_argument("--audio", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--device", choices=["cuda", "cpu"], default=None)
    add_whisper_opts(sp)
    sp.set_defaults(func=cmd_transcribe)

    # ── inspect ───────────────────────────────────────────────────────────── #
    sp = sub.add_parser("inspect", help="probe audio (+ esquema CSV opcional)")
    sp.add_argument("--audio", required=True)
    sp.add_argument("--transcripts", default=None)
    sp.set_defaults(func=cmd_inspect)

    # ── offsets (legacy) ──────────────────────────────────────────────────── #
    sp = sub.add_parser("offsets", help="[legacy] deriva offsets de chunks -> JSON")
    sp.add_argument("--chunks", required=True)
    sp.add_argument("--chunks-pattern", default="*.mp3", dest="chunks_pattern")
    sp.add_argument("--out", default="config/chunk_offsets.json")
    sp.set_defaults(func=cmd_offsets)

    # ── prep ──────────────────────────────────────────────────────────────── #
    sp = sub.add_parser("prep", help="audio -> WAV 16k mono")
    sp.add_argument("--audio", required=True)
    sp.add_argument("--out-wav", required=True, dest="out_wav")
    sp.set_defaults(func=cmd_prep)

    # ── diarize ───────────────────────────────────────────────────────────── #
    sp = sub.add_parser("diarize", help="WAV -> RTTM (NeMo)")
    sp.add_argument("--audio", required=True)
    sp.add_argument("--out", required=True)
    add_diar_opts(sp)
    sp.set_defaults(func=cmd_diarize)

    # ── align (legacy) ────────────────────────────────────────────────────── #
    sp = sub.add_parser("align", help="[legacy] RTTM + CSVs -> salida unificada")
    sp.add_argument("--rttm", required=True)
    sp.add_argument("--transcripts", required=True)
    sp.add_argument("--out", required=True)
    add_offset_opts(sp)
    sp.set_defaults(func=cmd_align)

    # ── export-txt ────────────────────────────────────────────────────────── #
    sp = sub.add_parser("export-txt", help="transcript_unificado.csv -> .txt tipo diálogo")
    sp.add_argument("--in", required=True, dest="input")
    sp.add_argument("--out", required=True)
    sp.add_argument("--title", default="Transcripción")
    sp.add_argument("--no-merge", action="store_true")
    sp.add_argument("--no-timestamps", action="store_true")
    sp.set_defaults(func=cmd_export_txt)

    # ── smoke ─────────────────────────────────────────────────────────────── #
    sp = sub.add_parser("smoke", help="pipeline sobre recorte corto (test rápido)")
    sp.add_argument("--audio", required=True)
    sp.add_argument("--out", default="output/smoke")
    sp.add_argument("--seconds", type=float, default=120.0)
    sp.add_argument("--start", type=float, default=0.0)
    add_diar_opts(sp)
    add_whisper_opts(sp)
    sp.set_defaults(func=cmd_smoke)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (audio_prep.AudioError, FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
