import audalign as ad
import pytest
import pickle

test_file_eig = "test_audio/test_shifts/Eigen-20sec.mp3"
test_file_eig2 = "test_audio/test_shifts/Eigen-song-base.mp3"
test_folder_eig = "test_audio/test_shifts/"


class TestAlign:
    ada = ad.Audalign(num_processors=4)

    @pytest.mark.smoke
    def test_align_fingerprint(self, tmpdir):
        result = self.ada.align("test_audio/test_shifts", tmpdir)
        assert result
        result = self.ada.align(
            "test_audio/test_shifts",
            tmpdir,
            write_extension=".wav",
        )
        assert result
        self.ada.pretty_print_results(result)

    def test_align_cor(self, tmpdir):
        result = self.ada.align(
            "test_audio/test_shifts",
            tmpdir,
            technique="correlation",
        )
        assert result

    def test_align_cor_options(self, tmpdir):
        result = self.ada.align(
            "test_audio/test_shifts",
            tmpdir,
            technique="correlation",
            cor_sample_rate=4000,
            filter_matches=0.3,  # might have to adjust this
            locality=30,
        )
        assert result
        self.ada.pretty_print_alignment(result, match_keys="match_info")

    def test_align_cor_spec(self, tmpdir):
        result = self.ada.align(
            "test_audio/test_shifts",
            tmpdir,
            technique="correlation_spectrogram",
        )
        assert result
        self.ada.pretty_print_alignment(result)

    def test_align_cor_spec_options(self, tmpdir):
        result = self.ada.align(
            "test_audio/test_shifts",
            tmpdir,
            technique="correlation_spectrogram",
            cor_sample_rate=4000,
            filter_matches=0.3,  # might have to adjust this
            locality=30,
            max_lags=10,
        )
        assert result
        self.ada.set_multiprocessing(False)
        result = self.ada.align(
            "test_audio/test_shifts",
            destination_path=tmpdir,
            technique="correlation_spectrogram",
            write_extension=".wav",
            cor_sample_rate=4000,
            filter_matches=0.3,  # might have to adjust this
            locality=30,
            max_lags=10,
        )
        assert result is not None
        self.ada.pretty_print_alignment(result)
        self.ada.set_multiprocessing(True)

    def test_align_vis(self, tmpdir):
        result = self.ada.align(
            "test_audio/test_shifts",
            tmpdir,
            technique="visual",
            volume_threshold=215,
            img_width=0.5,
        )
        assert result is not None
        self.ada.set_multiprocessing(False)
        result = self.ada.align(
            "test_audio/test_shifts",
            tmpdir,
            technique="visual",
            volume_threshold=215,
            img_width=0.5,
        )
        assert result is not None
        self.ada.set_multiprocessing(True)

    def test_align_badish_options(self, tmpdir):
        result = self.ada.align(
            "test_audio/test_shifts",
            tmpdir,
            write_extension="mov",
        )
        assert result

    @pytest.mark.xfail
    def test_align_bad_technique(self):
        self.ada.align("test_audio/test_shifts", technique="correlationion_bad")

    def test_align_files_fingerprints(self, tmpdir):
        result = self.ada.align_files(
            "test_audio/test_shifts/Eigen-20sec.mp3",
            "test_audio/test_shifts/Eigen-song-base.mp3",
            destination_path=tmpdir,
        )
        assert result

    def test_align_files_vis(self, tmpdir):
        result = self.ada.align_files(
            "test_audio/test_shifts/Eigen-20sec.mp3",
            "test_audio/test_shifts/Eigen-song-base.mp3",
            destination_path=tmpdir,
            technique="visual",
            volume_threshold=215,
            img_width=0.5,
        )
        assert result is not None
        self.ada.set_multiprocessing(False)
        result = self.ada.align_files(
            "test_audio/test_shifts/Eigen-20sec.mp3",
            "test_audio/test_shifts/Eigen-song-base.mp3",
            destination_path=tmpdir,
            technique="visual",
            volume_threshold=215,
            img_width=0.5,
        )
        assert result is not None
        self.ada.set_multiprocessing(True)

    def test_align_files_cor(self, tmpdir):
        result = self.ada.align_files(
            "test_audio/test_shifts/Eigen-20sec.mp3",
            "test_audio/test_shifts/Eigen-song-base.mp3",
            destination_path=tmpdir,
            write_extension=".wav",
            technique="correlation",
        )
        assert result is not None
        self.ada.set_multiprocessing(False)
        result = self.ada.align_files(
            "test_audio/test_shifts/Eigen-20sec.mp3",
            "test_audio/test_shifts/Eigen-song-base.mp3",
            destination_path=tmpdir,
            write_extension=".wav",
            technique="correlation",
        )
        assert result is not None
        self.ada.set_multiprocessing(True)


class TestTargetAlign:
    ada = ad.Audalign(num_processors=4)

    def test_target_align_vis(self, tmpdir):
        result = self.ada.target_align(
            "test_audio/test_shifts/Eigen-song-base.mp3",
            "test_audio/test_shifts",
            destination_path=tmpdir,
            technique="visual",
            img_width=0.5,
            volume_threshold=215,
        )
        assert result is not None
        self.ada.set_multiprocessing(False)
        result = self.ada.target_align(
            "test_audio/test_shifts/Eigen-song-base.mp3",
            "test_audio/test_shifts",
            destination_path=tmpdir,
            technique="visual",
            img_width=0.5,
            volume_threshold=215,
        )
        assert result is not None
        self.ada.set_multiprocessing(True)

    def test_target_align_vis_mse(self, tmpdir):
        result = self.ada.target_align(
            "test_audio/test_shifts/Eigen-song-base.mp3",
            "test_audio/test_shifts",
            destination_path=tmpdir,
            technique="visual",
            img_width=0.5,
            volume_threshold=215,
            calc_mse=True,
            start_end=(0, -1),
        )
        assert result

    @pytest.mark.xfail
    def test_target_align_bad_technique(self):
        self.ada.target_align(
            "test_audio/test_shifts/Eigen-song-base.mp3",
            "test_audio/test_shifts",
            technique="visual_bad",
        )

    def test_target_align_cor(self, tmpdir):
        result = self.ada.target_align(
            "test_audio/test_shifts/Eigen-song-base.mp3",
            "test_audio/test_shifts",
            destination_path=tmpdir,
            technique="correlation",
        )
        assert result

    def test_target_align_cor_spec(self, tmpdir):
        result = self.ada.target_align(
            "test_audio/test_shifts/Eigen-song-base.mp3",
            "test_audio/test_shifts",
            destination_path=tmpdir,
            technique="correlation_spectrogram",
        )
        assert result

    def test_target_align_fingerprints(self, tmpdir):
        result = self.ada.target_align(
            "test_audio/test_shifts/Eigen-song-base.mp3",
            "test_audio/test_shifts",
            destination_path=tmpdir,
            technique="fingerprints",
        )
        assert result


class TestFineAlign:
    ada = ad.Audalign(num_processors=4)

    with open("tests/align_test.pickle", "rb") as f:
        align_fing_results = pickle.load(f)
    align_fing_results = ada.recalc_shifts(align_fing_results)

    @pytest.mark.smoke
    def test_fine_align(self):
        result = self.ada.fine_align(
            self.align_fing_results,
        )
        assert result is not None

    def test_fine_align_spec(self):
        result = self.ada.fine_align(
            self.align_fing_results,
            technique="correlation_spectrogram",
        )
        assert result is not None
        self.ada.set_multiprocessing(False)
        result = self.ada.fine_align(
            self.align_fing_results,
            technique="correlation_spectrogram",
        )
        self.ada.set_multiprocessing(True)
        assert result is not None

    def test_fine_align_locality(self):
        result = self.ada.fine_align(
            self.align_fing_results,
            locality=10,
        )
        assert result is not None
        self.ada.pretty_print_alignment(result, match_keys="match_info")

    def test_fine_align_fingerprints(self, tmpdir):
        result = self.ada.fine_align(
            self.align_fing_results,
            technique="fingerprints",
            destination_path=tmpdir,
            locality=5,
            locality_filter_prop=0.5,
        )
        assert result is not None
        self.ada.pretty_print_alignment(result, match_keys="match_info")

    def test_fine_align_visual(self, tmpdir):
        result = self.ada.fine_align(
            self.align_fing_results,
            technique="visual",
            destination_path=tmpdir,
            volume_threshold=214,
            img_width=0.5,
        )
        assert result is not None
        self.ada.set_multiprocessing(False)
        result = self.ada.fine_align(
            self.align_fing_results,
            technique="visual",
            destination_path=tmpdir,
            volume_threshold=214,
            img_width=0.5,
        )
        assert result is not None
        self.ada.set_multiprocessing(True)
        self.ada.pretty_print_alignment(result, match_keys="fine_match_info")

    def test_fine_align_options(self, tmpdir):
        result = self.ada.fine_align(
            self.align_fing_results,
            destination_path=tmpdir,
            cor_sample_rate=8000,
            max_lags=5,
            match_index=1,
            write_extension=".ogg",
            filter_matches=0.1,
        )
        assert result is not None
        self.ada.pretty_print_results(result)


class TestRecalcWriteShifts:
    # TODO test recalc shifts and write from results
    ada = ad.Audalign(num_processors=2)

    with open("tests/align_test.pickle", "rb") as f:
        align_fing_results = pickle.load(f)
    align_fing_results = ada.recalc_shifts(align_fing_results)
    full_results = ada.fine_align(
        align_fing_results,
    )

    def test_recalc_shifts(self):
        temp_results = self.ada.recalc_shifts(self.align_fing_results)
        assert temp_results is not None

        temp_results = self.ada.recalc_shifts(self.full_results)
        assert temp_results is not None
        temp_results = self.ada.recalc_shifts(self.full_results, key="match_info")
        assert temp_results is not None
        temp_results = self.ada.recalc_shifts(
            self.full_results, key="only_fine_match_info"
        )
        assert temp_results is not None

    def test_recalc_shifts_indexes(self):
        temp_results = self.ada.recalc_shifts(self.align_fing_results, match_index=1)
        assert temp_results is not None

        temp_results = self.ada.recalc_shifts(self.full_results, match_index=1)
        assert temp_results is not None
        temp_results = self.ada.recalc_shifts(
            self.full_results, key="match_info", match_index=1
        )
        assert temp_results is not None
        temp_results = self.ada.recalc_shifts(
            self.full_results, key="only_fine_match_info", match_index=1
        )
        assert temp_results is not None

        temp_results = self.ada.recalc_shifts(
            self.full_results, match_index=1, fine_match_index=1
        )
        assert temp_results is not None
        temp_results = self.ada.recalc_shifts(
            self.full_results, key="match_info", match_index=1, fine_match_index=1
        )
        assert temp_results is not None
        temp_results = self.ada.recalc_shifts(
            self.full_results,
            key="only_fine_match_info",
            match_index=1,
            fine_match_index=1,
        )
        assert temp_results is not None

    def test_write_from_results(self, tmpdir):
        self.ada.write_shifts_from_results(self.full_results, test_folder_eig, tmpdir)
        self.ada.write_shifts_from_results(
            self.full_results, test_folder_eig, tmpdir, write_extension=".mp3"
        )

        # sources from original file location
        self.ada.write_shifts_from_results(
            self.full_results, None, tmpdir, write_extension=".mp3"
        )

        self.ada.write_shifts_from_results(
            self.full_results, "no errors just prints", tmpdir, write_extension=".mp3"
        )
