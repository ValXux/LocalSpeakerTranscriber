"""Tests del detector/normalizador de columnas para varios formatos de CSV."""
import pandas as pd
import pytest

from transcripcion.csv_loader import normalize_frame, parse_time, _read_frame


# --- formato real: nivel-segmento, headers en español, texto con comas ------ #
SPANISH_SEGMENT = (
    "Tiempo de inicio,Tiempo de finalización,Código de idioma,Confianza,Canal,"
    "Etiqueta de interlocutor,Transcripción\n"
    '0.000,28.160,es,,,0,"Hola, esto es una prueba, con comas."\n'
    '28.360,28.440,es,,,1," Ya."\n'
    '28.440,77.360,es,,,0," Segunda intervención del 0."\n'
)

# --- nivel-palabra, headers en inglés (otro export de Google STT) ----------- #
ENGLISH_WORD = (
    "word,start_time,end_time,speaker_tag\n"
    "Hello,0.0,0.5,1\n"
    "there,0.5,0.9,1\n"
    "how,1.0,1.3,2\n"
    "are,1.3,1.6,2\n"
    "you,1.6,1.9,2\n"
    "fine,2.0,2.4,1\n"
)

# --- nivel-segmento con tiempos tipo reloj ---------------------------------- #
CLOCK_SEGMENT = (
    "speaker,start,end,transcript\n"
    "A,0:00:01.000,0:00:03.500,uno\n"
    "B,0:01:00.000,0:01:02.000,dos\n"
)


def test_spanish_segment_level():
    segs, info = normalize_frame(_read_frame(SPANISH_SEGMENT), source="parte_000")
    assert info.level == "segment"
    assert info.time_format == "seconds"
    assert info.columns_detected["start"] == "Tiempo de inicio"
    assert info.columns_detected["speaker"] == "Etiqueta de interlocutor"
    assert info.columns_detected["text"] == "Transcripción"
    assert len(segs) == 3
    assert segs[0].start == 0.0 and segs[0].end == 28.16
    assert segs[0].speaker_local == "0"
    assert "comas" in segs[0].text
    assert segs[0].lang == "es"
    assert segs[0].source == "parte_000"


def test_english_word_level_grouping():
    segs, info = normalize_frame(_read_frame(ENGLISH_WORD), source="words")
    assert info.level == "word"
    # speakers consecutivos: 1,1 -> seg; 2,2,2 -> seg; 1 -> seg  => 3 segmentos
    assert len(segs) == 3
    assert segs[0].speaker_local == "1"
    assert segs[0].text == "Hello there"
    assert segs[0].start == 0.0 and segs[0].end == 0.9
    assert segs[1].speaker_local == "2"
    assert segs[1].text == "how are you"
    assert segs[1].start == 1.0 and segs[1].end == 1.9
    assert segs[2].text == "fine"


def test_clock_time_format():
    segs, info = normalize_frame(_read_frame(CLOCK_SEGMENT), source="clk")
    assert info.time_format == "clock"
    assert segs[0].start == 1.0 and segs[0].end == 3.5
    assert segs[1].start == 60.0 and segs[1].end == 62.0


def test_missing_columns_raises():
    bad = pd.DataFrame({"foo": ["a"], "bar": ["b"]})
    with pytest.raises(ValueError):
        normalize_frame(bad, source="bad")


@pytest.mark.parametrize("value,expected", [
    ("28.36", 28.36),
    ("28.36s", 28.36),
    (5, 5.0),
    ("0:00:28.360", 28.36),
    ("1:02", 62.0),
    ("1:00:00", 3600.0),
])
def test_parse_time(value, expected):
    assert parse_time(value) == pytest.approx(expected)
