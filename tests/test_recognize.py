import audalign as ad
import pickle
import pytest
import sys

test_file = "test_audio/testers/test.mp3"
test_file2 = "test_audio/testers/pink_noise.mp3"


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

    @pytest.mark.smoke
    def test_recognize_locality(self):
        assert self.ada.total_fingerprints > 0
        self.ada.set_accuracy(4)

        result = self.ada.recognize(test_file, locality=10)
        assert len(result) > 1

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

    def test_correcognize_no_return(self):
        results = self.ada.correcognize(
            test_file,
            test_file,
            filter_matches=2,
        )
        assert results is None

    def test_correcognize_directory(self):
        results = self.ada.correcognize_directory(
            test_file,
            "test_audio/testers/",
        )
        assert results

    def test_correcognize_directory_no_return(self):
        results = self.ada.correcognize_directory(
            test_file,
            "tests/",
        )
        assert results is None


class TestAlign:

    ada = ad.Audalign()

    with open("tests/align_test.pickle", "rb") as f:
        align_fing_results = pickle.load(f)

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

    def test_align_badish_options(self, tmpdir):
        result = self.ada.align(
            "test_audio/test_shifts",
            tmpdir,
            write_extension="mov",
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

    def test_align_files_cor(self, tmpdir):
        result = self.ada.align_files(
            "test_audio/test_shifts/Eigen-20sec.mp3",
            "test_audio/test_shifts/Eigen-song-base.mp3",
            destination_path=tmpdir,
            write_extension=".wav",
            technique="correlation",
        )
        assert result

    def test_target_align_fingerprint(self, tmpdir):
        result = self.ada.target_align(
            "test_audio/test_shifts/Eigen-song-base.mp3",
            "test_audio/test_shifts",
            destination_path=tmpdir,
        )
        assert result

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
