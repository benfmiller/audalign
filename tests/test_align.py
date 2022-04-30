import pickle

import audalign as ad
import pytest
from audalign import recognizers

test_file_eig = "test_audio/test_shifts/Eigen-20sec.mp3"
test_file_eig2 = "test_audio/test_shifts/Eigen-song-base.mp3"
test_folder_eig = "test_audio/test_shifts/"


class TestAlign:
    fingerprint_recognizer = ad.FingerprintRecognizer(
        load_fingerprints_file="tests/test_fingerprints.json"
    )

    @pytest.mark.smoke
    def test_align_fingerprint(self, tmpdir):
        result = ad.align("test_audio/test_shifts", tmpdir)
        assert result
        result = ad.align(
            "test_audio/test_shifts",
            tmpdir,
            write_extension=".wav",
            recognizer=self.fingerprint_recognizer,
        )
        assert result
        ad.pretty_print_results(result)

    def test_align_fingerprint_write_multi_channel(self, tmpdir):
        result = ad.align("test_audio/test_shifts", tmpdir)
        assert result
        result = ad.align(
            "test_audio/test_shifts",
            tmpdir,
            write_extension=".wav",
            write_multi_channel=True,
            recognizer=self.fingerprint_recognizer,
        )
        assert result
        ad.pretty_print_results(result)

    def test_align_cor(self, tmpdir):
        result = ad.align(
            "test_audio/test_shifts", tmpdir, recognizer=ad.CorrelationRecognizer()
        )
        assert result

    def test_align_cor_options(self, tmpdir):
        recognizer = ad.CorrelationRecognizer()
        recognizer.config.sample_rate = 4000
        recognizer.config.filter_matches = 0.3
        recognizer.config.locality = 30
        result = ad.align(
            "test_audio/test_shifts",
            tmpdir,
            recognizer=recognizer,
        )
        assert result
        ad.pretty_print_alignment(result, match_keys="match_info")

    def test_align_cor_spec(self, tmpdir):
        result = ad.align(
            "test_audio/test_shifts",
            tmpdir,
        )
        assert result
        ad.pretty_print_alignment(result)

    def test_align_cor_spec_options(self, tmpdir):
        recognizer = ad.CorrelationSpectrogramRecognizer()
        recognizer.config.sample_rate = 4000
        recognizer.config.filter_matches = 0.3
        recognizer.config.locality = 30
        recognizer.config.max_lags = 10
        result = ad.align(
            "test_audio/test_shifts",
            tmpdir,
            recognizer=recognizer,
        )
        assert result
        recognizer.config.multiprocessing = False
        result = ad.align(
            "test_audio/test_shifts",
            destination_path=tmpdir,
            write_extension=".wav",
            recognizer=recognizer,
        )
        assert result is not None
        ad.pretty_print_alignment(result)

    def test_align_vis(self, tmpdir):
        recognizer = ad.VisualRecognizer()
        recognizer.config.volume_threshold = 214
        recognizer.config.img_width = 0.5
        result = ad.align("test_audio/test_shifts", tmpdir, recognizer=recognizer)
        assert result is not None
        recognizer.config.multiprocessing = False
        result = ad.align("test_audio/test_shifts", tmpdir, recognizer=recognizer)
        assert result is not None

    def test_align_badish_options(self, tmpdir):
        result = ad.align(
            "test_audio/test_shifts",
            tmpdir,
            write_extension="mov",
        )
        assert result

    def test_align_load_fingerprints(self):
        recognizer = ad.FingerprintRecognizer(
            load_fingerprints_file="tests/test_fingerprints.json"
        )
        result = ad.align(
            "test_audio/test_shifts",
            recognizer=recognizer,
        )
        assert result


class TestAlignFiles:
    def test_align_files_fingerprints(self, tmpdir):
        result = ad.align_files(
            "test_audio/test_shifts/Eigen-20sec.mp3",
            "test_audio/test_shifts/Eigen-song-base.mp3",
            destination_path=tmpdir,
        )
        assert result

    def test_align_files_load_fingerprints(self):
        recognizer = ad.FingerprintRecognizer(
            load_fingerprints_file="tests/test_fingerprints.json"
        )
        result = ad.align_files(
            "test_audio/test_shifts/Eigen-20sec.mp3",
            "test_audio/test_shifts/Eigen-song-base.mp3",
            recognizer=recognizer,
        )
        assert result

    def test_align_files_vis(self, tmpdir):
        recognizer = ad.VisualRecognizer()
        recognizer.config.volume_threshold = 214
        recognizer.config.img_width = 0.5
        result = ad.align_files(
            "test_audio/test_shifts/Eigen-20sec.mp3",
            "test_audio/test_shifts/Eigen-song-base.mp3",
            destination_path=tmpdir,
            recognizer=recognizer,
        )
        assert result is not None
        recognizer.config.multiprocessing = False
        result = ad.align_files(
            "test_audio/test_shifts/Eigen-20sec.mp3",
            "test_audio/test_shifts/Eigen-song-base.mp3",
            destination_path=tmpdir,
            recognizer=recognizer,
        )
        assert result is not None

    def test_align_files_cor(self, tmpdir):
        recognizer = ad.CorrelationRecognizer()
        result = ad.align_files(
            "test_audio/test_shifts/Eigen-20sec.mp3",
            "test_audio/test_shifts/Eigen-song-base.mp3",
            destination_path=tmpdir,
            write_extension=".wav",
            recognizer=recognizer,
        )
        assert result is not None
        recognizer.config.multiprocessing = False
        result = ad.align_files(
            "test_audio/test_shifts/Eigen-20sec.mp3",
            "test_audio/test_shifts/Eigen-song-base.mp3",
            destination_path=tmpdir,
            write_extension=".wav",
            recognizer=recognizer,
        )
        assert result is not None


class TestTargetAlign:
    def test_target_align_vis(self, tmpdir):
        recognizer = ad.VisualRecognizer()
        recognizer.config.volume_threshold = 214
        recognizer.config.img_width = 0.5
        result = ad.target_align(
            "test_audio/test_shifts/Eigen-song-base.mp3",
            "test_audio/test_shifts",
            destination_path=tmpdir,
            recognizer=recognizer,
        )
        assert result is not None
        recognizer.config.multiprocessing = False
        result = ad.target_align(
            "test_audio/test_shifts/Eigen-song-base.mp3",
            "test_audio/test_shifts",
            destination_path=tmpdir,
            recognizer=recognizer,
        )
        assert result is not None

    def test_target_align_vis_mse(self, tmpdir):
        recognizer = ad.VisualRecognizer()
        recognizer.config.volume_threshold = 214
        recognizer.config.img_width = 0.5
        recognizer.config.calc_mse = True
        recognizer.config.start_end = (0, -1)
        result = ad.target_align(
            "test_audio/test_shifts/Eigen-song-base.mp3",
            "test_audio/test_shifts",
            destination_path=tmpdir,
            recognizer=recognizer,
        )
        assert result

    def test_target_align_cor(self, tmpdir):
        result = ad.target_align(
            "test_audio/test_shifts/Eigen-song-base.mp3",
            "test_audio/test_shifts",
            destination_path=tmpdir,
            recognizer=ad.CorrelationRecognizer(),
        )
        assert result

    def test_target_align_cor_spec(self, tmpdir):
        result = ad.target_align(
            "test_audio/test_shifts/Eigen-song-base.mp3",
            "test_audio/test_shifts",
            destination_path=tmpdir,
            recognizer=ad.CorrelationSpectrogramRecognizer(),
        )
        assert result

    def test_target_align_fingerprints(self, tmpdir):
        result = ad.target_align(
            "test_audio/test_shifts/Eigen-song-base.mp3",
            "test_audio/test_shifts",
            destination_path=tmpdir,
            recognizer=ad.FingerprintRecognizer(),
        )
        assert result

    def test_target_align_load_fingerprints(self):
        recognizer = ad.FingerprintRecognizer(
            load_fingerprints_file="tests/test_fingerprints.json"
        )
        result = ad.target_align(
            "test_audio/test_shifts/Eigen-song-base.mp3",
            "test_audio/test_shifts",
            recognizer=recognizer,
        )
        assert result


class TestFineAlign:
    with open("tests/align_test.pickle", "rb") as f:
        align_fing_results = pickle.load(f)
    align_fing_results = ad.recalc_shifts(align_fing_results)

    @pytest.mark.smoke
    def test_fine_align(self):
        result = ad.fine_align(
            self.align_fing_results,
        )
        assert result is not None

    def test_fine_align(self):
        result = ad.fine_align(
            self.align_fing_results,
            write_multi_channel=True,
        )
        assert result is not None

    def test_fine_align_spec(self):
        recognizer = ad.CorrelationSpectrogramRecognizer()
        result = ad.fine_align(
            self.align_fing_results,
            recognizer=recognizer,
        )
        assert result is not None
        recognizer.config.multiprocessing = False
        result = ad.fine_align(
            self.align_fing_results,
        )
        assert result is not None

    def test_fine_align_locality(self):
        recognizer = ad.CorrelationRecognizer()
        recognizer.config.locality = 10
        result = ad.fine_align(self.align_fing_results, recognizer=recognizer)
        assert result is not None
        ad.pretty_print_alignment(result, match_keys="match_info")

    def test_fine_align_fingerprints(self, tmpdir):
        recognizer = ad.FingerprintRecognizer()
        recognizer.config.locality = 5
        recognizer.config.locality_filter_prop = 0.5
        result = ad.fine_align(
            self.align_fing_results,
            destination_path=tmpdir,
            recognizer=recognizer,
        )
        assert result is not None
        ad.pretty_print_alignment(result, match_keys="match_info")

    def test_fine_align_load_fingerprints(self):
        recognizer = ad.FingerprintRecognizer(
            load_fingerprints_file="tests/test_fingerprints.json"
        )
        recognizer.config.locality = 5
        recognizer.config.locality_filter_prop = 0.5
        result = ad.fine_align(
            self.align_fing_results,
            recognizer=recognizer,
        )
        assert result is not None
        ad.pretty_print_alignment(result, match_keys="match_info")

    def test_fine_align_visual(self, tmpdir):
        recognizer = ad.VisualRecognizer()
        recognizer.config.volume_threshold = 210
        recognizer.config.img_width = 0.5
        result = ad.fine_align(
            self.align_fing_results,
            destination_path=tmpdir,
            recognizer=recognizer,
        )
        assert result is not None
        recognizer.config.multiprocessing = False
        result = ad.fine_align(
            self.align_fing_results,
            destination_path=tmpdir,
            recognizer=recognizer,
        )
        assert result is not None
        ad.pretty_print_alignment(result, match_keys="fine_match_info")

    def test_fine_align_options(self, tmpdir):
        recognizer = ad.CorrelationRecognizer()
        recognizer.config.sample_rate = 8000
        recognizer.config.max_lags = 5
        recognizer.config.filter_matches = 0.1
        result = ad.fine_align(
            self.align_fing_results,
            destination_path=tmpdir,
            match_index=1,
            write_extension=".ogg",
            recognizer=recognizer,
        )
        assert result is not None
        ad.pretty_print_results(result)


class TestRecalcWriteShifts:
    with open("tests/align_test.pickle", "rb") as f:
        align_fing_results = pickle.load(f)
    align_fing_results = ad.recalc_shifts(align_fing_results)
    full_results = ad.fine_align(
        align_fing_results,
    )

    def test_recalc_shifts(self):
        temp_results = ad.recalc_shifts(self.align_fing_results)
        assert temp_results is not None

        temp_results = ad.recalc_shifts(self.full_results)
        assert temp_results is not None
        temp_results = ad.recalc_shifts(self.full_results, key="match_info")
        assert temp_results is not None
        temp_results = ad.recalc_shifts(self.full_results, key="only_fine_match_info")
        assert temp_results is not None

    def test_recalc_shifts_indexes(self):
        temp_results = ad.recalc_shifts(self.align_fing_results, match_index=1)
        assert temp_results is not None

        temp_results = ad.recalc_shifts(self.full_results, match_index=1)
        assert temp_results is not None
        temp_results = ad.recalc_shifts(
            self.full_results, key="match_info", match_index=1
        )
        assert temp_results is not None
        temp_results = ad.recalc_shifts(
            self.full_results, key="only_fine_match_info", match_index=1
        )
        assert temp_results is not None

        temp_results = ad.recalc_shifts(
            self.full_results, match_index=1, fine_match_index=1
        )
        assert temp_results is not None
        temp_results = ad.recalc_shifts(
            self.full_results, key="match_info", match_index=1, fine_match_index=1
        )
        assert temp_results is not None
        temp_results = ad.recalc_shifts(
            self.full_results,
            key="only_fine_match_info",
            match_index=1,
            fine_match_index=1,
        )
        assert temp_results is not None

    def test_write_from_results(self, tmpdir):
        ad.write_shifts_from_results(self.full_results, test_folder_eig, tmpdir)
        ad.write_shifts_from_results(
            self.full_results, test_folder_eig, tmpdir, write_extension=".mp3"
        )

        # sources from original file location
        ad.write_shifts_from_results(
            self.full_results, None, tmpdir, write_extension=".mp3"
        )

        ad.write_shifts_from_results(
            self.full_results, "no errors just prints", tmpdir, write_extension=".mp3"
        )
