import audalign as ad
import pytest
import os


def test_always_true():
    assert True


class TestFingerprinter:

    test_file = "test_audio/testers/test.mp3"

    @pytest.mark.smoke
    def test_initialization(self):

        ada_fing = ad.FingerprintRecognizer()
        assert ada_fing.total_fingerprints == 0

        ada2 = ad.FingerprintRecognizer(
            load_fingerprints_file="tests/test_fingerprints.json"
        )
        assert ada2.total_fingerprints > 0
        assert len(ada2.fingerprinted_files) > 0

    def test_filter_duplicates(self):
        ada1 = ad.FingerprintRecognizer()

        ada1.load_fingerprinted_files("tests/test_fingerprints.json")
        a = len(ada1.file_names)
        b = ada1.total_fingerprints
        c = len(ada1.fingerprinted_files)
        ada1.load_fingerprinted_files("tests/test_fingerprints.json")
        assert a == len(ada1.file_names)
        assert b == ada1.total_fingerprints
        assert c == len(ada1.fingerprinted_files)

        ada1.fingerprinted_files.extend(ada1.fingerprinted_files)
        ada1.total_fingerprints += ada1.total_fingerprints
        ada1.file_names.extend(ada1.file_names)
        ada1.filter_duplicates()
        assert a == len(ada1.file_names)
        assert b == ada1.total_fingerprints
        assert c == len(ada1.fingerprinted_files)

    def test_clear(self):
        ada = ad.FingerprintRecognizer(
            load_fingerprints_file="tests/test_fingerprints.json"
        )
        assert len(ada.fingerprinted_files) > 0
        ada.clear_fingerprints()
        assert len(ada.fingerprinted_files) == 0
        assert len(ada.file_names) == 0
        assert ada.total_fingerprints == 0

    def test_set_accuracy(self):
        config = ad.config.fingerprint.FingerprintConfig()
        assert config.get_accuracy() == 2
        assert config.default_amp_min == 65
        config.set_accuracy(3)
        config.set_accuracy(4)
        config.set_accuracy(1)
        assert config.get_accuracy() == 1
        assert config.default_amp_min == 80

        try:
            config.set_accuracy(0)
            assert False
        except ValueError as e:
            pass
        assert config.get_accuracy() == 1

    def test_write_and_load(self):
        ada = ad.FingerprintRecognizer(
            load_fingerprints_file="tests/test_fingerprints.json"
        )
        assert len(ada.file_names) > 0
        ada.save_fingerprinted_files("test_save_fingerprints.json")
        ada.save_fingerprinted_files("test_save_fingerprints.pickle")
        ada.save_fingerprinted_files("test_no_write.txt")  # doesn't write anything

        ada.clear_fingerprints()
        assert len(ada.file_names) == 0
        ada.load_fingerprinted_files("test_save_fingerprints.pickle")
        assert len(ada.file_names) > 0
        ada.clear_fingerprints()
        assert len(ada.file_names) == 0
        ada.load_fingerprinted_files("test_save_fingerprints.json")
        assert len(ada.file_names) > 0
        ada.load_fingerprinted_files("tests/test_audalign.py")  # Not Loaded
        ada.load_fingerprinted_files("file_not_there.json")

    def test_get_metadata(self):
        metatdata = ad.get_metadata(file_path=self.test_file)
        assert metatdata != {}

    def test_write_processed_file(self, tmpdir):
        ad.write_processed_file(self.test_file, tmpdir.join("test.wav"))


class TestFilehandler:

    test_file = "test_audio/testers/test.mp3"

    def test_read(self):
        array, _ = ad.filehandler.read(self.test_file, sample_rate=None)
        assert len(array) > 0

    def test_get_aud_dir(self):
        file_list = ad.filehandler.get_audio_files_directory("tests")
        assert len(file_list) == 0

    def test_write_shifted_file(self, tmpdir):
        ad.write_shifted_file(self.test_file, tmpdir.join("place.mp3"), 5)


class TestUniformLevel:
    test_eig_folder = "test_audio/test_shifts/"
    test_eig = "test_audio/test_shifts/Eigen-song-base.mp3"
    test_eig20 = "test_audio/test_shifts/Eigen-20sec.mp3"

    def test_uniform_level_dir(self, tmpdir):
        ad.uniform_level_directory(self.test_eig_folder, tmpdir)

    def test_uniform_level_dir_average(self, tmpdir):
        ad.uniform_level_directory(self.test_eig_folder, tmpdir, mode="average")

    def test_uniform_level_file(self, tmpdir):
        ad.uniform_level_file(self.test_eig, tmpdir)
        ad.uniform_level_file(self.test_eig, os.path.join(tmpdir, "whatever_file.mp3"))

    def test_uniform_level_file_average(self, tmpdir):
        ad.uniform_level_file(self.test_eig, tmpdir, mode="average")
        ad.uniform_level_file(
            self.test_eig, os.path.join(tmpdir, "whatever_file.mp3"), mode="average"
        )


class TestRemoveNoise:
    test_file = "test_audio/testers/test.mp3"

    def test_remove_noise_directory(self, tmpdir):
        ad.remove_noise_directory(
            "test_audio/testers", "test_audio/testers/pink_noise.mp3", 10, 30, tmpdir
        )

    def test_remove_noise_directory_single_process(self, tmpdir):
        ad.config.multiprocessing = False
        ad.remove_noise_directory(
            "test_audio/testers", "test_audio/testers/pink_noise.mp3", 10, 30, tmpdir
        )

    def test_remove_noise(self, tmpdir):
        ad.remove_noise_file(
            self.test_file,
            10,
            20,
            tmpdir.join("test.mp3"),
        )

        ad.remove_noise_file(
            self.test_file,
            1,
            3,
            tmpdir.join("test.mov"),
            alt_noise_filepath="test_audio/testers/pink_noise.mp3",
        )

        ad.remove_noise_file(self.test_file, 10, 20, tmpdir, write_extension="wav")

    @pytest.mark.xfail
    def test_remove_noise_bad_file(self):
        ad.remove_noise_file(
            "SillyFile.mp3",
        )


class TestStartEnd:
    test_file = "test_audio/testers/test.mp3"

    def test_start(self, tmpdir):
        ad.convert_audio_file(
            self.test_file, tmpdir.join("test_temp.mp3"), start_end=(0, 0)
        )
        ad.convert_audio_file(
            self.test_file,
            tmpdir.join("test_temp.mp3"),
            start_end=(10, 0),
        )
        ad.convert_audio_file(
            self.test_file,
            tmpdir.join("test_temp.mp3"),
            start_end=(1000, 0),
        )

    def test_end(self, tmpdir):
        ad.convert_audio_file(
            self.test_file,
            tmpdir.join("test_temp.mp3"),
            start_end=(0, 10),
        )
        ad.convert_audio_file(
            self.test_file,
            tmpdir.join("test_temp.mp3"),
            start_end=(0, -10),
        )
        ad.convert_audio_file(
            self.test_file,
            tmpdir.join("test_temp.mp3"),
            start_end=(0, 1000),
        )

    def test_both(self, tmpdir):
        ad.convert_audio_file(
            self.test_file,
            tmpdir.join("test_temp.mp3"),
            start_end=(5, 10),
        )
        ad.convert_audio_file(
            self.test_file,
            tmpdir.join("test_temp.mp3"),
            start_end=(5, -10),
        )

    def test_bounds_checks(self):
        try:
            ad.filehandler.read(self.test_file, start_end=(5, 4))
            assert False  # should have raised value error
        except ValueError:
            pass
        try:
            ad.filehandler.read(self.test_file, start_end=(-2, 0))
            assert False  # should have raised value error
        except ValueError:
            pass
        array, _ = ad.filehandler.read(self.test_file, start_end=(0, -10000))
        assert len(set(array)) == 1
