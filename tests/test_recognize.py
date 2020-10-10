import audalign as ad
import pytest

test_file = "audio_files/TestAudio/test.wav"

def test_recognize():

    ada = ad.Audalign()
    ada.fingerprint_file(test_file)
    assert ada.total_fingerprints > 0

    ada.fingerprinted_files[0][0] = "different"
    ada.file_names[0] = "different"

    result = ada.recognize(test_file)
    assert len(result) > 1

    result2 = ada.recognize(
        "audio_files/TestAudio/pink_noise.wav", filter_matches=3
    )
    assert not result2