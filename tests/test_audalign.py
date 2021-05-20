import audalign as ad
import pytest


def test_always_true():
    assert True


class TestObject:

    test_file = "test_audio/testers/test.mp3"

    @pytest.mark.smoke
    def test_initialization(self):

        ada = ad.Audalign()
        assert ada.total_fingerprints == 0

        ada2 = ad.Audalign("tests/test_fingerprints.json")
        assert ada2.total_fingerprints > 0
        assert len(ada2.fingerprinted_files) > 0

    def test_filter_duplicates(self):
        ada1 = ad.Audalign()

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
        ada = ad.Audalign("tests/test_fingerprints.json")
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
        ada = ad.Audalign("tests/test_fingerprints.json")
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

    def test_get_metadata(self):
        metatdata = ad.Audalign.get_metadata(file_path=self.test_file)
        assert metatdata != {}


class TestFilehandler:

    ada = ad.Audalign()
    test_file = "test_audio/testers/test.mp3"

    def test_read(self):
        array, _ = ad.filehandler.read(self.test_file, sample_rate=None)
        assert len(array) > 0

    def test_get_aud_dir(self):
        file_list = ad.filehandler.get_audio_files_directory("tests")
        assert len(file_list) == 0

    def test_shift_write_files(self, tmpdir):
        ada = ad.Audalign()
        ada.write_shifted_file(self.test_file, tmpdir.join("place.mp3"), 5)


class TestRemoveNoise:
    test_file = "test_audio/testers/test.mp3"

    def test_remove_noise_directory(self, tmpdir):
        ada = ad.Audalign()
        ada.remove_noise_directory(
            "test_audio/testers", "test_audio/testers/pink_noise.mp3", 10, 30, tmpdir
        )

    def test_remove_noise_directory_single_process(self, tmpdir):
        ada = ad.Audalign(multiprocessing=False)
        ada.remove_noise_directory(
            "test_audio/testers", "test_audio/testers/pink_noise.mp3", 10, 30, tmpdir
        )

    def test_remove_noise(self, tmpdir):
        ad.Audalign.remove_noise_file(
            self.test_file,
            10,
            20,
            tmpdir.join("test.mp3"),
        )

        ad.Audalign.remove_noise_file(
            self.test_file,
            1,
            3,
            tmpdir.join("test.mov"),
            alt_noise_filepath="test_audio/testers/pink_noise.mp3",
        )

    @pytest.mark.xfail
    def test_remove_noise_bad_file(self):
        ad.Audalign.remove_noise_file(
            "SillyFile.mp3",
        )


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

    def test_fingerprint_bad_file(self):
        ada2 = ad.Audalign()
        # both should print something and be just fine
        ada2.fingerprint_file("filenot_even_there.txt")
        ada2.fingerprint_file("requirements.txt")
        assert len(ada2.fingerprinted_files) == 0
        assert len(ada2.file_names) == 0
        assert ada2.total_fingerprints == 0


class TestStartEnd:
    test_file = "test_audio/testers/test.mp3"

    ada = ad.Audalign()

    def test_start(self, tmpdir):
        self.ada.convert_audio_file(
            self.test_file, tmpdir.join("test_temp.mp3"), start_end=(0, 0)
        )
        self.ada.convert_audio_file(
            self.test_file,
            tmpdir.join("test_temp.mp3"),
            start_end=(10, 0),
        )
        self.ada.convert_audio_file(
            self.test_file,
            tmpdir.join("test_temp.mp3"),
            start_end=(1000, 0),
        )

    def test_end(self, tmpdir):
        self.ada.convert_audio_file(
            self.test_file,
            tmpdir.join("test_temp.mp3"),
            start_end=(0, 10),
        )
        self.ada.convert_audio_file(
            self.test_file,
            tmpdir.join("test_temp.mp3"),
            start_end=(0, -10),
        )
        self.ada.convert_audio_file(
            self.test_file,
            tmpdir.join("test_temp.mp3"),
            start_end=(0, 1000),
        )

    def test_both(self, tmpdir):
        self.ada.convert_audio_file(
            self.test_file,
            tmpdir.join("test_temp.mp3"),
            start_end=(5, 10),
        )
        self.ada.convert_audio_file(
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
