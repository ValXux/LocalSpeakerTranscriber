"""Tests de alineación por solapamiento: casos sintéticos con resultado esperado."""
import pytest

from src.align import Aligner, assign_speaker, UNKNOWN
from src.rttm import RttmSegment


def rttm():
    # speaker_0: [0,10] y [20,30] ; speaker_1: [10,20] ; speaker_2: [30,32]
    return [
        RttmSegment(0.0, 10.0, "speaker_0"),
        RttmSegment(10.0, 20.0, "speaker_1"),
        RttmSegment(20.0, 30.0, "speaker_0"),
        RttmSegment(30.0, 32.0, "speaker_2"),
    ]


def test_full_overlap_single_speaker():
    al = assign_speaker(1.0, 9.0, rttm())
    assert al.speaker_global == "speaker_0"
    assert al.overlap_confidence == pytest.approx(1.0)
    assert al.ambiguous is False


def test_partial_overlap_confidence():
    # segmento [8,12]: 2s con speaker_0 (8-10) y 2s con speaker_1 (10-12)
    al = assign_speaker(8.0, 12.0, rttm())
    # empate -> gana el primero rankeado; ambos 2s -> conf = 2/4 = 0.5
    assert al.overlap_confidence == pytest.approx(0.5)
    assert al.ambiguous is True
    assert {al.speaker_global, al.runner_up} == {"speaker_0", "speaker_1"}


def test_clear_winner_with_ambiguity():
    # segmento [7,11]: 3s speaker_0 (7-10), 1s speaker_1 (10-11)
    al = assign_speaker(7.0, 11.0, rttm())
    assert al.speaker_global == "speaker_0"
    assert al.overlap_confidence == pytest.approx(3.0 / 4.0)
    assert al.ambiguous is True
    assert al.runner_up == "speaker_1"
    assert al.runner_up_seconds == pytest.approx(1.0)


def test_no_overlap_is_unknown():
    al = assign_speaker(40.0, 45.0, rttm())
    assert al.speaker_global == UNKNOWN
    assert al.overlap_confidence == 0.0


def test_same_speaker_two_rttm_segments_sum():
    # segmento [25,35]: speaker_0 (25-30)=5s, speaker_2 (30-32)=2s
    al = assign_speaker(25.0, 35.0, rttm())
    assert al.speaker_global == "speaker_0"
    # dur=10 -> conf = 5/10 = 0.5
    assert al.overlap_confidence == pytest.approx(0.5)
    assert al.overlap_seconds == pytest.approx(5.0)


def test_empty_rttm():
    al = assign_speaker(0.0, 5.0, [])
    assert al.speaker_global == UNKNOWN


def test_aligner_reuse_many_segments():
    aligner = Aligner(rttm())
    assert aligner.assign(1, 2).speaker_global == "speaker_0"
    assert aligner.assign(11, 19).speaker_global == "speaker_1"
    assert aligner.assign(30.5, 31.5).speaker_global == "speaker_2"
