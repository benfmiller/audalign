import audalign as ad
import pytest

def test_align():
    ada = ad.Audalign()
    result = ada.align("audio_files/shifts", "test_alignment")
    assert result