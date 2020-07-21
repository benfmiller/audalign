import audalign as ad


def test_always_true():
    assert True


class TestInit:
    def test_initialization(self):

        ada = ad.Audalign()

        ada2 = ad.Audalign("all_audio.json")
        assert ada2.total_fingerprints > 0
        assert len(ada2.fingerprinted_files) > 0

    def test_fingerprint_file(self):
        ada = ad.Audalign()
        ada.fingerprint_file("audio_files/TestAudio/test.wav")
        assert ada.total_fingerprints > 0

    def test_fingerprint_directory(self):
        ada_multi = ad.Audalign()
        ada_multi.fingerprint_directory("audio_files/processed_audio")
        assert ada_multi.total_fingerprints > 0
        assert len(ada_multi.fingerprinted_files) > 0

        ada_single = ad.Audalign()
        ada_single.fingerprint_directory("audio_files/processed_audio")
        assert ada_single.total_fingerprints > 0
        assert len(ada_single.fingerprinted_files) > 0

    def test_recognize(self):

        ada = ad.Audalign()
        ada.fingerprint_file("audio_files/TestAudio/test.wav")
        assert ada.total_fingerprints > 0

        result = ada.recognize("audio_files/TestAudio/test.wav")
        assert len(result) > 1

        result2 = ada.recognize("audio_files/TestAudio/pink_noise.wav")
        assert not result2
