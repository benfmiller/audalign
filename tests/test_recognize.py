import audalign as ad
import pytest

test_file = "audio_files/TestAudio/test.wav"


class TestRecognize:
    @pytest.mark.smoke
    def test_recognize(self):

        ada = ad.Audalign("all_audio_panako.json")
        assert ada.total_fingerprints > 0

        ada.fingerprinted_files[0][0] = "different"
        ada.file_names[0] = "different"

        result = ada.recognize(test_file)
        assert len(result) > 1

        result2 = ada.recognize(
            "audio_files/TestAudio/pink_noise.wav", filter_matches=3
        )
        assert not result2

    @pytest.mark.smoke
    def test_visrecognize(self):
        ada = ad.Audalign()
        results = ada.visrecognize(
            test_file,
            test_file,
            img_width=0.5,
            volume_threshold=215,
        )
        assert results

    def test_visrecognize_options(self):
        ada = ad.Audalign()
        results = ada.visrecognize(
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
        assert results["match_info"]["test.wav"]["mse"][0] == 20000000.0

    def test_visrecognize_directory(self):
        ada = ad.Audalign()
        results = ada.visrecognize_directory(
            test_file,
            "audio_files/processed_audio",
            img_width=0.5,
            volume_threshold=215,
        )
        assert results


class TestAlign:
    @pytest.mark.smoke
    def test_align(self):
        ada = ad.Audalign()
        result = ada.align("test_alignment/test_shifts", "test_alignment")
        assert result
        result = ada.align(
            "test_alignment/test_shifts", "test_alignment", write_extension=".wav"
        )
        assert result

    def test_target_align_fingerprint(self):
        ada = ad.Audalign()
        result = ada.target_align(
            "test_alignment/test_shifts/Eigen-song-base.wav",
            "test_alignment/test_shifts",
            destination_path="test_alignment",
        )
        assert result

    def test_target_align_vis(self):
        ada = ad.Audalign()
        result = ada.target_align(
            "test_alignment/test_shifts/Eigen-song-base.wav",
            "test_alignment/test_shifts",
            destination_path="test_alignment",
            use_fingerprints=False,
            img_width=0.5,
        )
        assert result
