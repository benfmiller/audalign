import audalign as ad
import pytest
import os


def test_always_true():
    assert True


class TestObject:

    test_file = "test_audio/test.wav"

    @pytest.mark.smoke
    def test_initialization(self):

        ada = ad.Audalign()
        assert ada.total_fingerprints == 0

        ada2 = ad.Audalign("all_audio.json")
        assert ada2.total_fingerprints > 0
        assert len(ada2.fingerprinted_files) > 0

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

    def test_clear(self):
        ada = ad.Audalign("all_audio.json")
        assert len(ada.fingerprinted_files) > 0
        ada.clear_fingerprints()
        assert len(ada.fingerprinted_files) == 0
        assert len(ada.file_names) == 0
        assert ada.total_fingerprints == 0

    def test_set_accuracy(self):
        ada = ad.Audalign()
        assert ada.get_accuracy() == 2
        assert ad.fingerprint.default_amp_min == 65
        ada.set_accuracy(3)
        ada.set_accuracy(4)
        ada.set_accuracy(1)
        assert ada.get_accuracy() == 1
        assert ad.fingerprint.default_amp_min == 80

    def test_set_num_processors(self):
        ada = ad.Audalign(num_processors=1)
        assert ada.num_processors == 1
        ada.set_num_processors(80)
        assert ada.num_processors == 80

    def test_freq_threshold(self):
        ada = ad.Audalign(freq_threshold=0)
        ada.set_freq_threshold(200)
        assert ad.fingerprint.threshold == 200

    def test_write_and_load(self):
        ada = ad.Audalign("all_audio_panako.json")
        assert len(ada.file_names) > 0
        ada.save_fingerprinted_files("test_save_fingerprints.json")
        ada.save_fingerprinted_files("test_save_fingerprints.pickle")
        ada.clear_fingerprints()
        assert len(ada.file_names) == 0
        ada.load_fingerprinted_files("test_save_fingerprints.pickle")
        assert len(ada.file_names) > 0
        ada.clear_fingerprints()
        assert len(ada.file_names) == 0
        ada.load_fingerprinted_files("test_save_fingerprints.json")
        assert len(ada.file_names) > 0


class TestRemoveNoise:
    test_file = "test_audio/testers/test.wav"
    if not os.path.isdir("test_alignment"):
        os.mkdir("test_alignment")

    def test_remove_noise_directory(self):
        ada = ad.Audalign()
        ada.remove_noise_directory(
            "test_audio/testers",
            "test_audio/testers/pink_noise.wav",
            10,
            30,
            "test_alignment",
        )

    def test_remove_noise(self):
        ad.Audalign.remove_noise_file(
            self.test_file,
            10,
            20,
            "test_alignment/test.wav",
        )

        ad.Audalign.remove_noise_file(
            self.test_file,
            1,
            3,
            "test_alignment/test.wav",
            alt_noise_filepath="test_audio/testers/pink_noise.wav",
        )


class TestFingerprinting:
    test_file = "test_audio/testers/test.wav"

    def test_fingerprint_file(self):
        ada = ad.Audalign()
        ada.fingerprint_file(self.test_file)
        ada.fingerprint_file(self.test_file, set_file_name="Sup", plot=False)
        assert ada.total_fingerprints > 0
        assert ada.file_names[0] == "test.wav"
        assert len(ada.fingerprinted_files) == 1

        ada.clear_fingerprints()
        assert ada.total_fingerprints == 0

        ada.fingerprint_file(self.test_file, set_file_name="Sup")
        assert ada.file_names[0] == "Sup"

    def test_fingerprint_file_hash_styles(self):
        ada = ad.Audalign(hash_style="base")
        ada.fingerprint_file(self.test_file)
        assert ada.total_fingerprints > 0
        ada.clear_fingerprints()

        ada.set_hash_style("panako")
        ada.fingerprint_file(self.test_file)
        assert ada.total_fingerprints > 0
        ada.clear_fingerprints()

        ada.set_hash_style("panako_mod")
        ada.fingerprint_file(self.test_file)
        assert ada.total_fingerprints > 0
        ada.clear_fingerprints()

        ada.set_hash_style("base_three")
        ada.fingerprint_file(self.test_file)
        assert ada.total_fingerprints > 0

    @pytest.mark.smoke
    def test_fingerprint_directory_multiprocessing(self):
        ada_multi = ad.Audalign()
        ada_multi.fingerprint_directory("test_audio/testers")
        assert ada_multi.total_fingerprints > 0
        assert len(ada_multi.fingerprinted_files) > 0

    def test_fingerprint_directory_single(self):
        ada_single = ad.Audalign(multiprocessing=False)
        ada_single.fingerprint_directory("test_audio/testers")
        assert ada_single.total_fingerprints > 0
        assert len(ada_single.fingerprinted_files) > 0
