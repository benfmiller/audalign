import audalign as ad
import pytest


def test_always_true():
    assert True


class TestInit:

    test_file = "audio_files/TestAudio/test.wav"

    def test_initialization(self):

        ada = ad.Audalign()

        ada2 = ad.Audalign("all_audio.json")
        assert ada2.total_fingerprints > 0
        assert len(ada2.fingerprinted_files) > 0

    def test_fingerprint_file(self):
        ada = ad.Audalign()
        ada.fingerprint_file(self.test_file)
        ada.fingerprint_file(
            "audio_files/TestAudio/test.wav", set_file_name="Sup", plot=False
        )
        assert ada.total_fingerprints > 0
        assert ada.file_names[0] == "test.wav"
        assert len(ada.fingerprinted_files) == 1

        ada.clear_fingerprints()
        assert ada.total_fingerprints == 0

        ada.fingerprint_file(self.test_file, set_file_name="Sup")
        assert ada.file_names[0] == "Sup"

    def test_fingerprint_directory(self):
        ada_multi = ad.Audalign()
        ada_multi.fingerprint_directory("audio_files/processed_audio")
        assert ada_multi.total_fingerprints > 0
        assert len(ada_multi.fingerprinted_files) > 0

        ada_single = ad.Audalign()
        ada_single.fingerprint_directory("audio_files/processed_audio")
        assert ada_single.total_fingerprints > 0
        assert len(ada_single.fingerprinted_files) > 0

    def test_filter_duplicates(self):
        ada1 = ad.Audalign()

        ada1.load_fingerprinted_files("all_audio_panako.json")
        a = len(ada1.file_names)
        b = ada1.total_fingerprints
        c = len(ada1.fingerprinted_files)
        ada1.load_fingerprinted_files("all_audio_panako.json")
        ada1.filter_duplicates()
        assert a == len(ada1.file_names)
        assert b == ada1.total_fingerprints
        assert c == len(ada1.fingerprinted_files)
        