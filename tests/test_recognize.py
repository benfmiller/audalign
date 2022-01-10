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

    def test_fingerprint_file(self):
        fingerprint_recognizer = ad.FingerprintRecognizer()
        fingerprint_recognizer.config.set_accuracy(1)
        fingerprint_recognizer.fingerprint_file(self.test_file)
        fingerprint_recognizer.fingerprint_file(self.test_file, set_file_name="Sup")
        assert fingerprint_recognizer.total_fingerprints > 0
        assert fingerprint_recognizer.file_names[0] == "test.mp3"
        assert len(fingerprint_recognizer.fingerprinted_files) == 1

        fingerprint_recognizer.clear_fingerprints()
        assert fingerprint_recognizer.total_fingerprints == 0

        fingerprint_recognizer.fingerprint_file(self.test_file, set_file_name="Sup")
        assert fingerprint_recognizer.file_names[0] == "Sup"

    def test_fingerprint_file_hash_styles(self):
        fingerprint_recognizer = ad.FingerprintRecognizer()

        fingerprint_recognizer.clear_fingerprints()
        fingerprint_recognizer.config.set_accuracy(1)
        fingerprint_recognizer.config.set_hash_style("base")
        fingerprint_recognizer.fingerprint_file(self.test_file)
        assert fingerprint_recognizer.total_fingerprints > 0
        fingerprint_recognizer.clear_fingerprints()

        fingerprint_recognizer.config.set_hash_style("panako")
        fingerprint_recognizer.fingerprint_file(self.test_file)
        assert fingerprint_recognizer.total_fingerprints > 0
        fingerprint_recognizer.clear_fingerprints()

        fingerprint_recognizer.config.set_hash_style("panako_mod")
        fingerprint_recognizer.fingerprint_file(self.test_file)
        assert fingerprint_recognizer.total_fingerprints > 0
        fingerprint_recognizer.clear_fingerprints()

        fingerprint_recognizer.config.set_hash_style("base_three")
        fingerprint_recognizer.fingerprint_file(self.test_file)
        assert fingerprint_recognizer.total_fingerprints > 0

    def test_fingerprint_bad_hash_styles(self):
        fingerprint_recognizer = ad.FingerprintRecognizer()
        try:
            fingerprint_recognizer.config.set_hash_style("bad_hash_style")
            assert False
        except ValueError as e:
            pass
        assert fingerprint_recognizer.config.hash_style == "panako_mod"
        fingerprint_recognizer.config.hash_style = "bad_hash_style"
        fingerprint_recognizer.fingerprint_file(self.test_file)
        assert len(fingerprint_recognizer.fingerprinted_files) == 0
        assert len(fingerprint_recognizer.file_names) == 0
        assert fingerprint_recognizer.total_fingerprints == 0

    @pytest.mark.smoke
    def test_fingerprint_directory_multiprocessing(self):
        fingerprint_recognizer = ad.FingerprintRecognizer()

        fingerprint_recognizer.clear_fingerprints()
        fingerprint_recognizer.config.multiprocessing = True
        fingerprint_recognizer.fingerprint_directory("test_audio/testers")
        assert fingerprint_recognizer.total_fingerprints > 0
        assert len(fingerprint_recognizer.fingerprinted_files) > 0

    def test_fingerprint_directory_single(self):
        fingerprint_recognizer = ad.FingerprintRecognizer()

        fingerprint_recognizer.config.multiprocessing = False
        fingerprint_recognizer.fingerprint_directory("test_audio/testers")
        assert fingerprint_recognizer.total_fingerprints > 0
        assert len(fingerprint_recognizer.fingerprinted_files) > 0

    def test_fingerprint_directory_not_there_already_done(self):
        fingerprint_recognizer = ad.FingerprintRecognizer()

        fingerprint_recognizer.fingerprint_directory("test_audio/testers")
        fingerprint_recognizer.load_fingerprinted_files("tests/test_fingerprints.json")
        fingerprint_recognizer.fingerprint_directory("test_audio/test_shifts")

    def test_fingerprint_bad_file(self):
        fingerprint_recognizer = ad.FingerprintRecognizer()
        assert len(fingerprint_recognizer.fingerprinted_files) == 0
        assert len(fingerprint_recognizer.file_names) == 0
        assert fingerprint_recognizer.total_fingerprints == 0

        # both should print something and be just fine
        fingerprint_recognizer.fingerprint_file("filenot_even_there.txt")
        fingerprint_recognizer.fingerprint_file("requirements.txt")
        assert len(fingerprint_recognizer.fingerprinted_files) == 0
        assert len(fingerprint_recognizer.file_names) == 0
        assert fingerprint_recognizer.total_fingerprints == 0


class TestRecognize:

    fingerprint_recognizer = ad.FingerprintRecognizer(
        load_fingerprints_file="tests/test_fingerprints.json"
    )

    @pytest.mark.smoke
    def test_recognize(self):
        assert self.fingerprint_recognizer.total_fingerprints > 0
        self.fingerprint_recognizer.config.set_accuracy(4)

        result = ad.recognize(test_file, recognizer=self.fingerprint_recognizer)
        assert len(result) > 1

        self.fingerprint_recognizer.config.filter_matches = 20000
        result2 = ad.recognize(test_file, recognizer=self.fingerprint_recognizer)
        self.fingerprint_recognizer.config.filter_matches = (
            ad.config.fingerprint.FingerprintConfig.filter_matches
        )
        assert not result2

    def test_recognize_fingerprint(self):
        ada2 = ad.FingerprintRecognizer()
        ada2.config.set_accuracy(1)

        ada2.fingerprint_file(test_file_eig)
        result = ada2.recognize(test_file_eig2)
        assert result

    def test_recognize_max_lags(self):
        _max_lags = 4
        self.fingerprint_recognizer.config.max_lags = 4
        results = self.fingerprint_recognizer.recognize(test_file_eig)
        self.fingerprint_recognizer.config.max_lags = (
            ad.config.fingerprint.FingerprintConfig.max_lags
        )

        offset_seconds = results["match_info"][os.path.basename(test_file_eig2)][
            "offset_seconds"
        ]
        assert min(offset_seconds) < _max_lags
        assert max(offset_seconds) < _max_lags

    @pytest.mark.smoke
    def test_recognize_locality(self):
        assert self.fingerprint_recognizer.total_fingerprints > 0
        self.fingerprint_recognizer.config.set_accuracy(4)

        self.fingerprint_recognizer.config.locality = 10
        result = ad.recognize(test_file, recognizer=self.fingerprint_recognizer)
        assert len(result) > 1
        ad.pretty_print_recognition(result)
        self.fingerprint_recognizer.config = ad.config.fingerprint.FingerprintConfig()

    def test_recognize_locality_max_lags(self):
        _max_lags = 4
        self.fingerprint_recognizer.config.max_lags = _max_lags
        self.fingerprint_recognizer.config.locality = 10
        results = ad.recognize(test_file_eig, recognizer=self.fingerprint_recognizer)

        offset_seconds = results["match_info"][os.path.basename(test_file_eig2)][
            "offset_seconds"
        ]
        assert min(offset_seconds) < _max_lags
        assert max(offset_seconds) < _max_lags
        self.fingerprint_recognizer.config = ad.config.fingerprint.FingerprintConfig()

    @pytest.mark.smoke
    def test_visrecognize(self):
        recognizer = ad.VisualRecognizer()
        recognizer.config.img_width = 0.5
        recognizer.config.volume_threshold = 200
        results = ad.recognize(
            test_file,
            test_file,
            recognizer=recognizer,
        )
        assert results

    def test_visrecognize_single_threaded(self):

        recognizer = ad.VisualRecognizer()
        recognizer.config.img_width = 0.5
        recognizer.config.volume_threshold = 200
        recognizer.config.multiprocessing = False
        recognizer.config.calc_mse = True
        results = ad.recognize(
            test_file,
            test_file,
            recognizer=recognizer,
        )
        assert results

    def test_visrecognize_options(self):
        recognizer = ad.VisualRecognizer()
        recognizer.config.img_width = 0.5
        recognizer.config.volume_threshold = 200
        recognizer.config.volume_floor = 100
        recognizer.config.vert_scaling = 0.8
        recognizer.config.horiz_scaling = 0.8

        results = ad.recognize(
            test_file,
            test_file,
            recognizer=recognizer,
        )
        assert results
        assert results["match_info"]["test.mp3"]["mse"][0] == 20000000.0

    def test_visrecognize_directory(self):
        recognizer = ad.VisualRecognizer()
        recognizer.config.img_width = 0.5
        recognizer.config.volume_threshold = 182
        results = ad.recognize(
            test_file,
            "test_audio/testers/",
            recognizer=recognizer,
        )
        assert results

    @pytest.mark.smoke
    def test_correcognize(self):
        recognizer = ad.CorrelationRecognizer()
        # recognizer.config.filter_matches = None
        results = ad.recognize(
            test_file,
            test_file,
            recognizer=recognizer,
        )
        assert results

    def test_correcognize_locality(self):
        recognizer = ad.CorrelationRecognizer()
        recognizer.config.locality = 10
        # recognizer.config.filter_matches = None
        results = ad.recognize(
            test_file,
            test_file,
            recognizer=recognizer,
        )
        assert results

    def test_correcognize_no_return(self):
        recognizer = ad.CorrelationRecognizer()
        recognizer.config.filter_matches = 2
        results = ad.recognize(
            test_file,
            test_file,
            recognizer=recognizer,
        )
        assert results is None

    def test_correcognize_directory_locality(self):
        recognizer = ad.CorrelationRecognizer()
        recognizer.config.locality = 10
        results = ad.recognize(
            test_file,
            "test_audio/testers/",
            recognizer=recognizer,
        )
        assert results

    def test_correcognize_directory(self):
        recognizer = ad.CorrelationRecognizer()
        results = ad.recognize(
            test_file,
            "test_audio/testers/",
            recognizer=recognizer,
        )
        assert results

    def test_correcognize_directory_single_threaded(self):
        recognizer = ad.CorrelationRecognizer()
        recognizer.config.multiprocessing = False
        results = ad.recognize(
            test_file,
            "test_audio/testers/",
            recognizer=recognizer,
        )
        assert results

    def test_correcognize_max_lags(self):
        _max_lags = 4
        recognizer = ad.CorrelationRecognizer()
        recognizer.config.max_lags = _max_lags
        results = ad.recognize(
            test_file_eig,
            test_file_eig2,
            recognizer=recognizer,
        )
        assert results
        offset_seconds = results["match_info"][os.path.basename(test_file_eig2)][
            "offset_seconds"
        ]
        assert min(offset_seconds) < _max_lags
        assert max(offset_seconds) < _max_lags

    def test_correcognize_locality_max_lags(self):
        _max_lags = 4
        recognizer = ad.CorrelationRecognizer()
        recognizer.config.max_lags = _max_lags
        recognizer.config.locality = 10
        results = ad.recognize(
            test_file_eig,
            test_file_eig2,
            recognizer=recognizer,
        )
        assert results
        offset_seconds = results["match_info"][os.path.basename(test_file_eig2)][
            "offset_seconds"
        ]
        assert min(offset_seconds) < _max_lags
        assert max(offset_seconds) < _max_lags

    def test_correcognize_directory_no_return(self):
        recognizer = ad.CorrelationRecognizer()
        results = ad.recognize(
            test_file,
            "tests/",
            recognizer=recognizer,
        )
        assert results is None

    def test_correcognize_spectrogram(self):
        recognizer = ad.CorrelationSpectrogramRecognizer()
        results = ad.recognize(
            test_file_eig,
            test_file_eig2,
            recognizer=recognizer,
        )
        assert results
        ad.pretty_print_results(results)

    def test_correcognize_spectrogram_locality(self):
        recognizer = ad.CorrelationSpectrogramRecognizer()
        recognizer.config.locality = 20
        results = ad.recognize(
            test_file_eig,
            test_file_eig2,
            recognizer=recognizer,
        )
        assert results

    def test_correcognize_spectrogram_directory(self):
        recognizer = ad.CorrelationSpectrogramRecognizer()
        results = ad.recognize(
            test_file_eig,
            test_folder_eig,
            recognizer=recognizer,
        )
        assert results

    def test_correcognize_spectrogram_max_lags(self):
        _max_lags = 4
        recognizer = ad.CorrelationSpectrogramRecognizer()
        recognizer.config.max_lags = _max_lags
        results = ad.recognize(test_file_eig, test_file_eig2, recognizer=recognizer)
        assert results
        offset_seconds = results["match_info"][os.path.basename(test_file_eig2)][
            "offset_seconds"
        ]
        assert min(offset_seconds) < _max_lags
        assert max(offset_seconds) < _max_lags

    def test_correcognize_spectrogram_locality_max_lags(self):
        _max_lags = 4
        recognizer = ad.CorrelationSpectrogramRecognizer()
        recognizer.config.max_lags = _max_lags
        recognizer.config.locality = 10
        results = ad.recognize(
            test_file_eig,
            test_file_eig2,
            recognizer=recognizer,
        )
        assert results
        offset_seconds = results["match_info"][os.path.basename(test_file_eig2)][
            "offset_seconds"
        ]
        assert min(offset_seconds) < _max_lags
        assert max(offset_seconds) < _max_lags