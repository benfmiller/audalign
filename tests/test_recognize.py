import audalign as ad
import pytest

test_file = "audio_files/TestAudio/test.wav"


@pytest.mark.smoke
def test_recognize():

    ada = ad.Audalign("all_audio_panako.json")
    assert ada.total_fingerprints > 0

    ada.fingerprinted_files[0][0] = "different"
    ada.file_names[0] = "different"

    result = ada.recognize(test_file)
    assert len(result) > 1

    result2 = ada.recognize("audio_files/TestAudio/pink_noise.wav", filter_matches=3)
    assert not result2


def test_visrecognize():
    ada = ad.Audalign()
    results = ada.visrecognize(
        test_file,
        test_file,
        img_width=0.5,
        volume_threshold=215,
    )
    assert results


def test_visrecognize_directory():
    ada = ad.Audalign()
    results = ada.visrecognize_directory(
        test_file,
        "audio_files/processed_audio",
        img_width=0.5,
        volume_threshold=215,
    )
    assert results
