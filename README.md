# LocalSpeakerTranscriber

Pipeline local para convertir un audio en una transcripcion con hablantes
separados. El flujo principal usa:

- `faster-whisper` para transcribir el audio.
- NVIDIA NeMo para diarizacion: VAD MarbleNet, embeddings TitaNet-large y
  clustering espectral.
- Alineacion temporal para asignar cada segmento de texto a un `speaker_global`.

No envia el audio a servicios externos. La primera ejecucion descarga modelos
preentrenados de Whisper/NeMo y requiere internet.

## Estado recomendado

Este proyecto queda preparado para correr en Windows nativo desde PowerShell,
sin WSL.

Entorno objetivo:

- Windows 11 64-bit.
- NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM.
- Driver NVIDIA instalado en Windows.
- `ffmpeg` y `ffprobe` instalados y visibles en `PATH`.
- `uv` instalado.
- Python 3.11 dentro de `.venv-win`.

Importante: aunque Windows tenga Python global 3.14, no lo uses para este
stack. PyTorch, NeMo y faster-whisper deben instalarse en el entorno aislado
`.venv-win` con Python 3.11.

## Instalacion en Windows nativo

Desde PowerShell, entra al proyecto:

```powershell
cd D:\Transcripcion
```

Instalacion automatica:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
.\.venv-win\Scripts\Activate.ps1
```

El script:

- Verifica `uv`, `ffmpeg`, `ffprobe` y `nvidia-smi`.
- Instala Python 3.11 con `uv` si falta.
- Crea `.venv-win`.
- Instala PyTorch/Torchaudio CUDA.
- Instala el resto de dependencias desde `requirements.txt`.
- Verifica que el CLI cargue.

Instalacion manual equivalente:

```powershell
uv python install 3.11
uv venv --python 3.11 .venv-win
.\.venv-win\Scripts\Activate.ps1

uv pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt
```

Verifica el entorno:

```powershell
python --version
where ffmpeg
where ffprobe
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -m src.cli --help
```

Si necesitas instalar sin GPU NVIDIA:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1 -Torch cpu
```

## Uso principal: audio directo

Coloca tu audio localmente, por ejemplo:

```text
data/audio/mi_audio.mp3
```

`data/` esta ignorado por Git para evitar publicar datos reales.

Primero corre un smoke test corto:

```powershell
python -m src.cli smoke `
  --audio data/audio/mi_audio.mp3 `
  --seconds 120 `
  --out output/smoke `
  --model small `
  --language es `
  --batch-size 4
```

Luego corre el audio completo:

```powershell
python -m src.cli run `
  --audio data/audio/mi_audio.mp3 `
  --out output `
  --model medium `
  --language es `
  --batch-size 8
```

Si sabes cuantas personas hablan, fuerza ese numero para mejorar el clustering:

```powershell
python -m src.cli run `
  --audio data/audio/mi_audio.mp3 `
  --out output `
  --model medium `
  --language es `
  --num-speakers 5 `
  --batch-size 8
```

Opciones utiles:

- `--model tiny|base|small|medium|large-v2|large-v3`: calidad y velocidad de
  Whisper. En una RTX 3050 de 4 GB, empieza con `small`.
- `--language es`: evita deteccion automatica si ya sabes el idioma.
- `--device cuda|cpu`: por defecto el sistema elige CUDA si esta disponible.
- `--batch-size 4` o `8`: recomendable en GPU de 4 GB.
- `--num-speakers N`: modo oracle, fuerza el numero de hablantes.
- `--max-num-speakers 20`: limite superior cuando el numero se detecta solo.

## Interfaz web

Con el entorno activado:

```powershell
python app.py
```

Abre:

```text
http://127.0.0.1:7860
```

La interfaz Gradio permite subir un audio, escoger modelo Whisper, idioma,
numero de hablantes y `batch_size`. Devuelve TXT, CSV y JSON.

## Salidas

El pipeline escribe en `output/`:

- `audio_16k.wav`: audio convertido a WAV mono 16 kHz.
- `diarization.rttm`: segmentos de diarizacion con `speaker_global`.
- `transcript_unificado.csv`: salida tabular principal.
- `transcript_unificado.json`: salida con metadatos extra.
- `reporte.md`: resumen de hablantes, duraciones y advertencias.
- `transcripcion.txt`: texto legible tipo dialogo.

Columnas principales del CSV:

```text
speaker_global,start_global,end_global,text,source_csv,speaker_local_original,overlap_confidence
```

En el flujo principal, `source_csv` y `speaker_local_original` quedan vacios
porque el texto viene de Whisper, no de CSVs externos.

## Comandos disponibles

```powershell
python -m src.cli --help
python -m src.cli run --help
python -m src.cli smoke --help
```

Subcomandos:

- `run`: pipeline completo con audio directo.
- `smoke`: recorta un audio y corre el pipeline sobre pocos segundos.
- `transcribe`: solo Whisper, genera segmentos con timestamps.
- `prep`: convierte audio a WAV mono 16 kHz.
- `diarize`: solo diarizacion NeMo, genera RTTM.
- `export-txt`: convierte `transcript_unificado.csv` a TXT.
- `inspect`: inspecciona audio y, opcionalmente, CSVs legacy.
- `offsets`: genera offsets para CSVs legacy por chunks.
- `align`: modo legacy, alinea RTTM existente con CSVs externos.

## Modo legacy: CSVs de Google STT

El flujo antiguo se conserva para casos donde ya tienes CSVs de Google
Speech-to-Text con timestamps y etiquetas locales de hablante. En ese modo,
NeMo solo provee `speaker_global` consistente y el texto sale de los CSVs.

Ejemplo:

```powershell
python -m src.cli align `
  --rttm output/diarization.rttm `
  --transcripts partido/transcripts/csv `
  --offsets config/chunk_offsets.json `
  --out output
```

`config/chunk_offsets.json` solo aplica a este modo legacy. Sirve cuando los
CSVs vienen de chunks separados y cada archivo empieza en tiempo `0`.

Para regenerar offsets desde chunks:

```powershell
python -m src.cli offsets `
  --chunks data/audio/chunks `
  --out config/chunk_offsets.json
```

## Tests

Los tests unitarios no requieren NeMo ni GPU:

```powershell
.\.venv-win\Scripts\Activate.ps1
python -m pytest -q
```

Tambien valida que el CLI cargue correctamente:

```powershell
python -m src.cli --help
python -m src.cli run --help
python -m src.cli smoke --help
```

## Troubleshooting

- `.\.venv\Scripts\python.exe` no existe: ese `.venv` es de WSL/Linux. Usa
  `.venv-win`.
- `python --version` muestra `3.14`: activa `.venv-win` o ejecuta
  `.\.venv-win\Scripts\python.exe`.
- `CUDA out of memory`: baja `--batch-size` a `4`, usa `--model small`, y
  cierra aplicaciones que esten usando VRAM.
- Sin GPU NVIDIA: usa `--device cpu` solo para pruebas cortas. Diarizar audios
  largos en CPU puede tardar horas.
- Primera ejecucion lenta: Whisper/NeMo descargan modelos y caches.
- `ffmpeg` o `ffprobe` no existe: instala ffmpeg para Windows y confirma con
  `where ffmpeg` y `where ffprobe`.
- `OMP: Error #15` o `libiomp5md.dll already initialized`: el paquete activa en
  Windows `KMP_DUPLICATE_LIB_OK=TRUE` al iniciar `src` para permitir que
  Torch/CTranslate2/NeMo convivan en el mismo proceso.
- `PicklingError` de `SpeechLabelEntity` en NeMo: en Windows el codigo fuerza
  `num_workers=0` para evitar multiprocessing en el DataLoader.
- `torch.cuda.is_available()` da `False`: revisa `nvidia-smi`, el driver NVIDIA
  de Windows y que instalaste PyTorch/Torchaudio con el indice CUDA.
- Si NeMo falla especificamente por compatibilidad Windows, conserva el resto
  del pipeline y evalua cambiar solo el backend de diarizacion.

## WSL como fallback

El script anterior de WSL sigue en `scripts/setup_wsl.sh`, pero ya no es el
camino principal. Usalo solo si Windows nativo no puede instalar o ejecutar
NeMo correctamente.

## Estructura

```text
src/
  app.py          interfaz Gradio
  cli.py          comandos del pipeline
  pipeline.py     audio -> diarizacion -> transcripcion -> salida
  audio_prep.py   ffprobe/ffmpeg y conversion a WAV 16 kHz mono
  diarize.py      NeMo ClusteringDiarizer -> RTTM
  transcribe.py   faster-whisper -> segmentos con timestamps
  align.py        asignacion por maximo solapamiento temporal
  consolidate.py  CSV/JSON/reporte
  export_txt.py   TXT legible tipo dialogo
  csv_loader.py   soporte legacy para CSVs de Google STT
  offsets.py      offsets legacy por chunks
  rttm.py         parser RTTM
config/
  diarizer.yaml       configuracion NeMo
  chunk_offsets.json  solo modo legacy
examples/
  datos ficticios pequenos para documentar formatos
tests/
  pruebas unitarias
```

## Autor
- Valentin Fernandez