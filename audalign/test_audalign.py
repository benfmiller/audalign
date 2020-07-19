import audalign as ad


def test_always_true():
    assert True


class TestInit:
    def test_initialization(self):

        ada = ad.Audalign()

    def test_fingerprint_file(self):
        ada = ad.Audalign()
        ada.fingerprint_file("audio_files/TestAudio/test.wav")
        assert ada.total_fingerprints > 0
