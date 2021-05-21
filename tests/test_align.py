import audalign as ad
import pytest
import pickle


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

    def test_align_cor(self, tmpdir):
        result = self.ada.align(
            "test_audio/test_shifts", tmpdir, technique="correlation"
        )
        assert result
        result = self.ada.align(
            "test_audio/test_shifts",
            tmpdir,
            technique="correlation",
            filter_matches=0.3,  # might have to adjust this
        )
        assert result

    def test_align_vis(self, tmpdir):
        result = self.ada.align(
            "test_audio/test_shifts",
            tmpdir,
            technique="visual",
            volume_threshold=215,
            img_width=0.5,
        )
        assert result

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
        assert result

    def test_align_files_cor(self, tmpdir):
        result = self.ada.align_files(
            "test_audio/test_shifts/Eigen-20sec.mp3",
            "test_audio/test_shifts/Eigen-song-base.mp3",
            destination_path=tmpdir,
            write_extension=".wav",
            technique="correlation",
        )
        assert result


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
        assert result

    def test_target_align_vis_mse(self, tmpdir):
        result = self.ada.target_align(
            "test_audio/test_shifts/Eigen-song-base.mp3",
            "test_audio/test_shifts",
            destination_path=tmpdir.join("emptydir/"),
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

    @pytest.mark.smoke
    def test_fine_align(self):
        result = self.ada.fine_align(
            self.align_fing_results,
        )
        assert result is not None

    def test_fine_align_fingerprints(self, tmpdir):
        result = self.ada.fine_align(
            self.align_fing_results,
            technique="fingerprints",
            destination_path=tmpdir,
            locality=5,
            locality_filter_prop=0.5,
        )
        assert result is not None

    def test_fine_align_visual(self, tmpdir):
        result = self.ada.fine_align(
            self.align_fing_results,
            technique="visual",
            destination_path=tmpdir,
            volume_threshold=214,
            img_width=0.5,
        )
        assert result is not None

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