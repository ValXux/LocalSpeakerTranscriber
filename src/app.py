"""Interfaz Gradio para el pipeline self-contained de transcripción + diarización.

Uso:
    python -m src.app
    python app.py          (desde raíz del proyecto)
"""
from __future__ import annotations

import threading
import tempfile
from pathlib import Path

import gradio as gr  # type: ignore

# NeMo + CUDA no son thread-safe: serializar llamadas al pipeline
_LOCK = threading.Lock()

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "diarizer.yaml"


def _run(audio_file, model_size, language, num_speakers_val, batch_size, progress=gr.Progress()):
    if audio_file is None:
        return "No se subió ningún audio.", None, None, None

    progress(0.0, desc="Iniciando...")

    out_dir = Path(tempfile.mkdtemp(prefix="transcripcion_"))

    lang = None if language == "auto" else language
    num_spk = None if int(num_speakers_val) == 0 else int(num_speakers_val)
    audio_name = Path(audio_file).stem

    def on_progress(pct: float, msg: str) -> None:
        progress(pct, desc=msg)

    with _LOCK:
        from .pipeline import run_full_pipeline
        result = run_full_pipeline(
            audio_file,
            out_dir,
            whisper_model=model_size,
            language=lang,
            num_speakers=num_spk,
            device=None,
            batch_size=int(batch_size),
            config_path=str(_CONFIG_PATH),
            title=audio_name,
            progress_cb=on_progress,
        )

    paths = result["paths"]
    txt_content = paths["txt"].read_text(encoding="utf-8")

    return (
        txt_content,
        str(paths["csv"]),
        str(paths["txt"]),
        str(paths["json"]),
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Transcripción automática con hablantes") as demo:
        gr.Markdown("# Transcripción automática con detección de hablantes")
        gr.Markdown(
            "Subí un audio y el sistema detecta a los participantes y transcribe automáticamente. "
            "No se envía nada a servicios externos — todo corre localmente."
        )

        with gr.Row():
            # ── columna izquierda: inputs ──────────────────────────────────── #
            with gr.Column(scale=1, min_width=280):
                audio_input = gr.Audio(
                    label="Audio",
                    type="filepath",
                    sources=["upload"],
                )

                with gr.Accordion("Configuración", open=True):
                    model_dd = gr.Dropdown(
                        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                        value="medium",
                        label="Modelo Whisper",
                        info="large-v3 = mejor calidad, más lento",
                    )
                    lang_dd = gr.Dropdown(
                        choices=["auto", "es", "en", "pt", "fr", "de", "it"],
                        value="auto",
                        label="Idioma",
                        info="auto = detección automática",
                    )
                    num_spk_sl = gr.Slider(
                        minimum=0, maximum=10, value=0, step=1,
                        label="Número de hablantes",
                        info="0 = detección automática",
                    )
                    batch_sl = gr.Slider(
                        minimum=4, maximum=32, value=16, step=4,
                        label="Batch size NeMo",
                        info="Reducir a 4-8 si hay error de memoria GPU",
                    )

                run_btn = gr.Button("Transcribir", variant="primary", size="lg")

            # ── columna derecha: outputs ───────────────────────────────────── #
            with gr.Column(scale=2):
                output_text = gr.Textbox(
                    label="Transcripción",
                    lines=28,
                    placeholder="El resultado aparecerá aquí...",
                )
                with gr.Row():
                    dl_csv = gr.File(label="CSV", file_count="single")
                    dl_txt = gr.File(label="TXT", file_count="single")
                    dl_json = gr.File(label="JSON", file_count="single")

        run_btn.click(
            fn=_run,
            inputs=[audio_input, model_dd, lang_dd, num_spk_sl, batch_sl],
            outputs=[output_text, dl_csv, dl_txt, dl_json],
            show_progress="full",
        )

    return demo


def main() -> None:
    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
