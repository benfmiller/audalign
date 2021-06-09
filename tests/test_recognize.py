import audalign as ad
import os
import pytest

test_file = "test_audio/testers/test.mp3"
test_file2 = "test_audio/testers/pink_noise.mp3"
test_file_eig = "test_audio/test_shifts/Eigen-20sec.mp3"
test_file_eig2 = "test_audio/test_shifts/Eigen-song-base.mp3"
test_folder_eig = "test_audio/test_shifts/"


class TestFingerprinting:
    test_file = "test_audio/testers/test.mp3"
    ada = ad.Audalign()

    def test_fingerprint_file(self):
        self.ada.clear_fingerprints()
        self.ada.set_accuracy(1)
        self.ada.fingerprint_file(self.test_file)
        self.ada.fingerprint_file(self.test_file, set_file_name="Sup", plot=False)
        assert self.ada.total_fingerprints > 0
        assert self.ada.file_names[0] == "test.mp3"
        assert len(self.ada.fingerprinted_files) == 1

        self.ada.clear_fingerprints()
        assert self.ada.total_fingerprints == 0

        self.ada.fingerprint_file(self.test_file, set_file_name="Sup")
        assert self.ada.file_names[0] == "Sup"

    def test_fingerprint_file_hash_styles(self):
        self.ada.clear_fingerprints()
        self.ada.set_accuracy(1)
        self.ada.set_hash_style("base")
        self.ada.fingerprint_file(self.test_file)
        assert self.ada.total_fingerprints > 0
        self.ada.clear_fingerprints()

        self.ada.set_hash_style("panako")
        self.ada.fingerprint_file(self.test_file)
        assert self.ada.total_fingerprints > 0
        self.ada.clear_fingerprints()

        self.ada.set_hash_style("panako_mod")
        self.ada.fingerprint_file(self.test_file)
        assert self.ada.total_fingerprints > 0
        self.ada.clear_fingerprints()

        self.ada.set_hash_style("base_three")
        self.ada.fingerprint_file(self.test_file)
        assert self.ada.total_fingerprints > 0

    def test_fingerprint_bad_hash_styles(self):
        ada1 = ad.Audalign(hash_style="bad_hash_style")
        assert ada1.hash_style == "panako_mod"
        ada1.hash_style = "bad_hash_style"
        ada1.fingerprint_file(self.test_file)
        assert len(ada1.fingerprinted_files) == 0
        assert len(ada1.file_names) == 0
        assert ada1.total_fingerprints == 0

    @pytest.mark.smoke
    def test_fingerprint_directory_multiprocessing(self):
        self.ada.clear_fingerprints()
        self.ada.set_multiprocessing(True)
        self.ada.fingerprint_directory("test_audio/testers")
        assert self.ada.total_fingerprints > 0
        assert len(self.ada.fingerprinted_files) > 0

    def test_fingerprint_directory_single(self):
        self.ada.set_multiprocessing(False)
        self.ada.clear_fingerprints()
        self.ada.fingerprint_directory("test_audio/testers")
        assert self.ada.total_fingerprints > 0
        assert len(self.ada.fingerprinted_files) > 0

    def test_fingerprint_directory_not_there_already_done(self):
        self.ada.fingerprint_directory("test_audio/testers")
        self.ada.load_fingerprinted_files("tests/test_fingerprints.json")
        self.ada.fingerprint_directory("test_audio/test_shifts")

    def test_fingerprint_bad_file(self):
        ada2 = ad.Audalign()
        # both should print something and be just fine
        ada2.fingerprint_file("filenot_even_there.txt")
        ada2.fingerprint_file("requirements.txt")
        assert len(ada2.fingerprinted_files) == 0
        assert len(ada2.file_names) == 0
        assert ada2.total_fingerprints == 0


class TestRecognize:

    ada = ad.Audalign("tests/test_fingerprints.json")

    @pytest.mark.smoke
    def test_recognize(self):
        assert self.ada.total_fingerprints > 0
        self.ada.set_accuracy(4)

        result = self.ada.recognize(test_file)
        assert len(result) > 1

        result2 = self.ada.recognize(test_file, filter_matches=20000)
        assert not result2

    def test_recognize_fingerprint(self):
        ada2 = ad.Audalign()
        ada2.fingerprint_file(test_file)
        result = ada2.recognize(test_file2)
        assert result

    def test_recognize_max_lags(self):
        _max_lags = 4
        results = self.ada.recognize(test_file_eig, max_lags=4)

        offset_seconds = results["match_info"][os.path.basename(test_file_eig2)][
            "offset_seconds"
        ]
        assert min(offset_seconds) < _max_lags
        assert max(offset_seconds) < _max_lags

    @pytest.mark.smoke
    def test_recognize_locality(self):
        assert self.ada.total_fingerprints > 0
        self.ada.set_accuracy(4)

        result = self.ada.recognize(test_file, locality=10)
        assert len(result) > 1

    # @pytest.mark.skip(reason="not working for some reason")
    def test_recognize_locality_max_lags(self):
        _max_lags = 4
        results = self.ada.recognize(test_file_eig, max_lags=4, locality=10)

        offset_seconds = results["match_info"][os.path.basename(test_file_eig2)][
            "offset_seconds"
        ]
        assert min(offset_seconds) < _max_lags
        assert max(offset_seconds) < _max_lags

    @pytest.mark.smoke
    def test_visrecognize(self):
        results = self.ada.visrecognize(
            test_file,
            test_file,
            img_width=0.5,
            volume_threshold=215,
        )
        assert results

    def test_visrecognize_single_threaded(self):
        self.ada.set_multiprocessing(False)
        results = self.ada.visrecognize(
            test_file,
            test_file,
            img_width=0.5,
            volume_threshold=215,
            calc_mse=True,
        )
        self.ada.set_multiprocessing(True)
        assert results

    def test_visrecognize_options(self):
        results = self.ada.visrecognize(
            test_file,
            test_file,
            img_width=0.5,
            volume_threshold=216,
            volume_floor=100,
            vert_scaling=0.8,
            horiz_scaling=0.8,
            calc_mse=False,
        )
        assert results
        assert results["match_info"]["test.mp3"]["mse"][0] == 20000000.0

    def test_visrecognize_directory(self):
        results = self.ada.visrecognize_directory(
            test_file,
            "test_audio/testers/",
            img_width=0.5,
            volume_threshold=215,
        )
        assert results

    @pytest.mark.smoke
    def test_correcognize(self):
        results = self.ada.correcognize(
            test_file,
            test_file,
            filter_matches=None,  # sets to 0.5
        )
        assert results

    def test_correcognize(self):
        results = self.ada.correcognize(
            test_file,
            test_file,
            locality=10,
            filter_matches=None,  # sets to 0.5
        )
        assert results

    def test_correcognize_no_return(self):
        results = self.ada.correcognize(
            test_file,
            test_file,
            filter_matches=2,
        )
        assert results is None

    def test_correcognize_directory_locality(self):
        results = self.ada.correcognize_directory(
            test_file,
            "test_audio/testers/",
            locality=10,
        )
        assert results

    def test_correcognize_directory(self):
        results = self.ada.correcognize_directory(
            test_file,
            "test_audio/testers/",
        )
        assert results

    def test_correcognize_max_lags(self):
        _max_lags = 4
        results = self.ada.correcognize(
            test_file_eig,
            test_file_eig2,
            max_lags=_max_lags,
            filter_matches=0,
        )
        assert results
        offset_seconds = results["match_info"][os.path.basename(test_file_eig2)][
            "offset_seconds"
        ]
        assert min(offset_seconds) < _max_lags
        assert max(offset_seconds) < _max_lags

    def test_correcognize_locality_max_lags(self):
        _max_lags = 4
        results = self.ada.correcognize(
            test_file_eig,
            test_file_eig2,
            max_lags=_max_lags,
            locality=10,
            filter_matches=0,
        )
        assert results
        offset_seconds = results["match_info"][os.path.basename(test_file_eig2)][
            "offset_seconds"
        ]
        assert min(offset_seconds) < _max_lags
        assert max(offset_seconds) < _max_lags

    def test_correcognize_directory_no_return(self):
        results = self.ada.correcognize_directory(
            test_file,
            "tests/",
        )
        assert results is None

    def test_correcognize_spectrogram(self):
        results = self.ada.correcognize_spectrogram(
            test_file_eig,
            test_file_eig2,
        )
        assert results

    def test_correcognize_spectrogram_locality(self):
        results = self.ada.correcognize_spectrogram(
            test_file_eig,
            test_file_eig2,
            locality=10,
        )
        assert results

    def test_correcognize_spectrogram_directory(self):
        results = self.ada.correcognize_spectrogram_directory(
            test_file_eig,
            test_folder_eig,
        )
        assert results

    def test_correcognize_spectrogram_max_lags(self):
        _max_lags = 4
        results = self.ada.correcognize_spectrogram(
            test_file_eig,
            test_file_eig2,
            max_lags=_max_lags,
            filter_matches=0,
        )
        assert results
        offset_seconds = results["match_info"][os.path.basename(test_file_eig2)][
            "offset_seconds"
        ]
        assert min(offset_seconds) < _max_lags
        assert max(offset_seconds) < _max_lags

    def test_correcognize_spectrogram_locality_max_lags(self):
        _max_lags = 4
        results = self.ada.correcognize_spectrogram(
            test_file_eig,
            test_file_eig2,
            max_lags=_max_lags,
            filter_matches=0,
            locality=10,
        )
        assert results
        offset_seconds = results["match_info"][os.path.basename(test_file_eig2)][
            "offset_seconds"
        ]
        assert min(offset_seconds) < _max_lags
        assert max(offset_seconds) < _max_lags