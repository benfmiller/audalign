import json
import multiprocessing
import os
import pickle
from functools import partial

from pydub.utils import mediainfo

import audalign.align as align
import audalign.correcognize as correcognize
import audalign.filehandler as filehandler
import audalign.fingerprint as fingerprint
import audalign.recognize as recognize
import audalign.visrecognize as visrecognize


class Audalign:

    # Names that appear in match information
    CONFIDENCE = "confidence"
    MATCH_TIME = "match_time"
    OFFSET_SAMPLES = "offset_frames"
    OFFSET_SECS = "offset_seconds"
    LOCALITY = "locality_frames"
    LOCALITY_SECS = "locality_seconds"

    def __init__(
        self,
        *args,
        multiprocessing=True,
        num_processors=None,
        hash_style="panako_mod",
        accuracy=2,
        freq_threshold=100,
    ):
        """
        Constructs new audalign object

        hash style has four options. All fingerprints must be of the same hash style to match.

        'base' hash style consists of two peaks. Two frequencies and a time difference.
        Creates many matches but is insensitive to noise.

        'panako' hash style consists of three peaks. Two differences in frequency, two frequency
        bands, one time difference ratio. Creates few matches, very resistant to noise.

        'panako_mod' hash style consists of three peaks. Two differences in frequency and one
        time difference ratio. Creates less matches than base, more than panako. moderately
        resistant to noise

        'base_three' hash style consists of three peaks. Three frequencies and two time differences.

        multiprocessing is set to True by default

        There are four accuracy levels with 1 being the lowest accuracy but the fastest. 3 is the highest recommended.
        4 gives the highest accuracy, but can take several gigabytes of memory for a couple files.
        Accuracy settings are acheived by manipulations in fingerprinting variables.

        Args
            arg1 (str): Optional file path to load json or pickle file of already fingerprinted files
            multiprocessing (bool): option to turn off multiprocessing
            num_processors (int): number of processors or threads to use for multiprocessing. Uses all if not given
            hash_style (str): which hash style to use : ['base','panako_mod','panako', 'base_three']
            accuracy (int): which accuracy level 1-4
            threshold(int): filters fingerprints below threshold
            noisereduce(bool): runs noise reduce on audio
        """

        self.file_names = []
        self.fingerprinted_files = []
        self.total_fingerprints = 0

        if len(args) > 0:
            self.load_fingerprinted_files(args[0])

        self.set_num_processors(num_processors)
        self.set_multiprocessing(multiprocessing)
        self.set_freq_threshold(freq_threshold)
        self.set_hash_style(hash_style)
        self.set_accuracy(accuracy)

    def set_hash_style(self, hash_style: str) -> None:
        """Sets the hash style. Must be one of ["base", "panako", "panako_mod", "base_three"]

        Args:
            hash_style (str): Method to use for hashing of fingerprints
        """
        if hash_style not in ["base", "panako", "panako_mod", "base_three"]:
            print(
                'Hash style must be one of ["base", "panako", "panako_mod", "base_three"]'
            )
            return
        self.hash_style = hash_style

    def set_accuracy(self, accuracy: int) -> None:
        """
        Sets the accuracy level of audalign object

        There are four accuracy levels with 1 being the lowest accuracy but the fastest. 3 is the highest recommended.
        4 gives the highest accuracy, but can take several gigabytes of memory for a couple files.
        Accuracy settings are acheived by manipulations in fingerprinting variables.

        Specific values for accuracy levels were chosen semi-arbitrarily from experimentation to give a few good options.

        Args
            accuracy (int): which accuracy level: 1-4
        """
        if accuracy < 1 or accuracy > 4:
            print("Accuracy must be between 1 and 4")
            return
        self.accuracy = accuracy
        self._set_accuracy(accuracy)

    @staticmethod
    def _set_accuracy(accuracy: int) -> None:
        if accuracy == 1:
            fingerprint.default_fan_value = 15
            fingerprint.default_amp_min = 80
            fingerprint.min_hash_time_delta = 10
            fingerprint.max_hash_time_delta = 200
            fingerprint.peak_sort = True
        elif accuracy == 2:
            fingerprint.default_fan_value = 15
            fingerprint.default_amp_min = 65
            fingerprint.min_hash_time_delta = 10
            fingerprint.max_hash_time_delta = 200
            fingerprint.peak_sort = True
        elif accuracy == 3:
            fingerprint.default_fan_value = 40
            fingerprint.default_amp_min = 60
            fingerprint.min_hash_time_delta = 1
            fingerprint.max_hash_time_delta = 400
            fingerprint.peak_sort = True
        elif accuracy == 4:
            fingerprint.default_fan_value = 60
            fingerprint.default_amp_min = 55
            fingerprint.min_hash_time_delta = 1
            fingerprint.max_hash_time_delta = 2000
            fingerprint.peak_sort = True

    def get_accuracy(self) -> int:
        """Current Accuracy from 1-4

        Returns:
            [int]: Accuracy level
        """
        return self.accuracy

    def set_freq_threshold(self, threshold: int) -> None:
        """Sets minimum frequency threshold for fingerprint

        Args:
            threshold ([int]): [threshold]
        """
        self.freq_threshold = threshold
        self._set_freq_threshold(freq_threshold=threshold)

    @staticmethod
    def _set_freq_threshold(freq_threshold: int) -> None:
        fingerprint.threshold = freq_threshold

    def get_freq_threshold(self):
        """Returns the frequency threshold

        Returns:
            [int]: frequency threshold
        """
        return self.freq_threshold

    def set_multiprocessing(self, true_or_false: bool) -> None:
        """Sets to true for on or false for off

        Args:
            true_or_false (bool): [true on or false off]
        """
        self.multiprocessing = true_or_false

    def set_num_processors(self, num_processors: int) -> None:
        """Set to none to use all processors by default if multiprocessing is true

        Args:
            num_processors (int): number of processors to use or None for all of them
        """
        self.num_processors = num_processors

    def save_fingerprinted_files(self, filename: str) -> None:
        """
        Serializes fingerprinted files to json or pickle file

        Args:
            filename (str): file to load saved fingerprints from
        """

        data = [self.fingerprinted_files, self.total_fingerprints, self.file_names]
        if filename.split(".")[-1] == "pickle":
            with open(filename, "wb") as f:
                pickle.dump(data, f)
        elif filename.split(".")[-1] == "json":
            with open(filename, "w") as f:
                json.dump(data, f)
        else:
            print("File type must be either pickle or json")

    def load_fingerprinted_files(self, filename: str) -> None:
        """
        Loads/adds saved json or pickle file into current audalign object

        Args
            filename (str): must be either json or pickle extension

        Returns
        -------
        None
        """
        try:
            if filename.split(".")[-1] == "pickle":
                with open(filename, "rb") as f:
                    data = pickle.load(f)
            elif filename.split(".")[-1] == "json":
                with open(filename, "r") as f:
                    data = json.load(f)
            else:
                print("File type must be either pickle or json")
                return
            self.fingerprinted_files.extend(data[0])
            self.total_fingerprints += data[1]
            self.file_names.extend(data[2])
            self.filter_duplicates()
        except FileNotFoundError:
            print(f'"{filename}" not found')

    def filter_duplicates(self) -> None:
        """
        Removes copies of fingerprinted files with the same name
        """
        name_checker = set()
        i = 0
        while i < len(self.file_names):
            if self.file_names[i] in name_checker:
                self.total_fingerprints -= len(self.fingerprinted_files[i][1])
                self.fingerprinted_files.pop(i)
                self.file_names.pop(i)
            else:
                name_checker.add(self.file_names[i])
                i += 1

    def fingerprint_directory(
        self, path: str, plot=False, extensions=["*"], _file_audsegs: dict = None
    ) -> None:
        """
        Fingerprints all files in given directory and all subdirectories

        Args
            path (str): path to directory to be fingerprinted
            plot (boolean): if true, plots the peaks to be fingerprinted on a spectrogram
            extensions (list[str]): specify which extensions to fingerprint
            _file_audsegs (dict, None): For internal use with fine align

        Returns
        -------
        None
        """

        result = self._fingerprint_directory(
            path, plot=plot, extensions=extensions, _file_audsegs=_file_audsegs
        )

        if result:
            for processed_file in result:
                if (
                    processed_file[0] != None
                    and processed_file[0] not in self.file_names
                ):
                    self.fingerprinted_files.append(processed_file)
                    self.file_names.append(processed_file[0])
                    self.total_fingerprints += len(processed_file[1])

    def _fingerprint_directory(
        self,
        path: str,
        plot: bool = False,
        extensions: list = ["*"],
        _file_audsegs: dict = None,
    ):
        """
        Worker function for fingerprint_directory

        Fingerprints all files in given directory and all subdirectories

        Args
            path (str): path to directory to be fingerprinted
            plot (boolean): if true, plots the peaks to be fingerprinted on a spectrogram
            extensions (list[str]): specify which extensions to fingerprint

        Returns
        -------
        None
        """
        if type(path) == str:
            file_names = filehandler.find_files(path, extensions)
        elif type(path) == list:
            file_names = zip(path, ["_"] * len(path))
        elif path is None and _file_audsegs is not None:
            file_names = zip(_file_audsegs.keys(), ["_"] * len(_file_audsegs))

        filenames_to_fingerprint = []
        for filename, _ in file_names:  # finds all files to fingerprint
            file_name = os.path.basename(filename)
            if file_name in self.file_names:
                print(f"{file_name} already fingerprinted")
                continue
            filenames_to_fingerprint.append(filename)

        if len(filenames_to_fingerprint) == 0:
            print("Directory contains 0 files or could not be found")
            return

        if _file_audsegs is not None:
            filenames_to_fingerprint = [
                (filename, _file_audsegs[filename])
                for filename in filenames_to_fingerprint
            ]

        _fingerprint_worker_directory = partial(
            fingerprint._fingerprint_worker,
            hash_style=self.hash_style,
            plot=plot,
            accuracy=self.accuracy,
            freq_threshold=self.freq_threshold,
        )

        if self.multiprocessing == True:

            # Try to use the maximum amount of processes if not given.
            try:
                nprocesses = self.num_processors or multiprocessing.cpu_count()
            except NotImplementedError:
                nprocesses = 1
            else:
                nprocesses = 1 if nprocesses <= 0 else nprocesses

            with multiprocessing.Pool(nprocesses) as self.pool:

                result = self.pool.map(
                    _fingerprint_worker_directory, filenames_to_fingerprint
                )

                self.pool.close()
                self.pool.join()

        else:

            result = []

            for filename in filenames_to_fingerprint:
                file_name = os.path.basename(filename)
                if file_name in self.file_names:
                    print(f"{file_name} already fingerprinted, continuing...")
                    continue
                file_name, hashes = _fingerprint_worker_directory(filename)
                if file_name == None:
                    continue
                result.append([file_name, hashes])
        return result

    def fingerprint_file(
        self,
        file_path: str,
        start_end: tuple = None,
        set_file_name: str = None,
        plot: bool = False,
    ) -> None:
        """
        Fingerprints given file and adds to fingerprinted files

        Args
            file_path (str): path to word to be fingerprinted
            start_end (tuple(float, float), optional): Silences before and after start and end. (0, -1) Silences last second, (5.4, 0) silences first 5.4 seconds
            set_file_name (str): option to set file name manually rather than use file name in file_path
            plot (boolean): if true, plots the peaks to be fingerprinted on a spectrogram

        Returns
        -------
        None
        """
        file_name = os.path.basename(file_path)
        if file_name in self.file_names:
            print(f"{file_name} already fingerprinted")
            return None

        file_name, hashes = self._fingerprint_file(
            file_path, start_end=start_end, set_file_name=set_file_name, plot=plot
        )
        if file_name != None:
            self.fingerprinted_files.append([file_name, hashes])
            self.file_names.append(file_name)
            self.total_fingerprints += len(hashes)

    def _fingerprint_file(
        self,
        file_path: str,
        start_end: tuple = None,
        set_file_name: str = None,
        plot: bool = False,
    ):
        """
        Worker function for fingerprint_file

        Fingerprints given file and adds to fingerprinted files

        Args
            file_path (str): path to word to be fingerprinted
            start_end (tuple(float, float), optional): Silences before and after start and end. (0, -1) Silences last second, (5.4, 0) silences first 5.4 seconds
            set_file_name (str): option to set file name manually rather than use file name in file_path
            plot (boolean): if true, plots the peaks to be fingerprinted on a spectrogram

        Returns
        -------
        [file_name, hashes]
        """

        file_name, hashes = fingerprint._fingerprint_worker(
            file_path,
            self.hash_style,
            start_end=start_end,
            plot=plot,
            accuracy=self.accuracy,
            freq_threshold=self.freq_threshold,
        )
        file_name = set_file_name or file_name
        return [file_name, hashes]

    def recognize(
        self,
        file_path: str,
        filter_matches: int = 1,
        locality: float = None,
        locality_filter_prop: float = None,
        start_end: tuple = None,
        max_lags: float = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Recognizes given file against already fingerprinted files

        Offset describes duration that the recognized file aligns after the target file
        Does not recognize against files with same name and extention
        Locality option used to only return match results within certain second range

        Args
            file_path (str): file path of target file to recognize
            filter_matches (int): filters all matches lower than given argument, 1 is recommended
            locality (float): filters matches to only count within locality. In seconds
            locality_filter_prop (int, float,optional): within each offset, filters locality tuples by proportion of highest confidence to tuple confidence
            start_end (tuple(float, float), optional): Silences before and after start and end. (0, -1) Silences last second, (5.4, 0) silences first 5.4 seconds
            max_lags (float, optional): Maximum lags in seconds.

        Returns
        -------
        match_result : dict
            dictionary containing match time and match info

            or

            None : if no match
        """

        if locality_filter_prop is None or locality_filter_prop > 1.0:
            locality_filter_prop = 1.0

        return recognize.recognize(
            self,
            file_path=file_path,
            filter_matches=filter_matches,
            locality=locality,
            locality_filter_prop=locality_filter_prop,
            start_end=start_end,
            max_lags=max_lags,
            *args,
            **kwargs,
        )

    def correcognize(
        self,
        target_file_path: str,
        against_file_path: str,
        start_end_target: tuple = None,
        start_end_against: tuple = None,
        filter_matches: float = 0.5,
        match_len_filter: int = None,
        sample_rate: int = fingerprint.DEFAULT_FS,
        max_lags: float = None,
        plot: bool = False,
        **kwargs,
    ):
        """Uses cross correlation to find alignment

        Faster than visrecognize or recognize and more useful for amplitude
        based alignments

        Args:
            target_file_path (str): File to recognize
            against_file_path (str): File to recognize against
            start_end_target (tuple(float, float), optional): Silences before and after start and end. (0, -1) Silences last second, (5.4, 0) silences first 5.4 seconds
            start_end_against (tuple(float, float), optional): Silences before and after start and end. (0, -1) Silences last second, (5.4, 0) silences first 5.4 seconds
            filter_matches (float, optional): Filters based on confidence. Ranges between 0 and 1. Defaults to 0.5.
            match_len_filter (int, optional): Limits number of matches returned. Defaults to 30.
            sample_rate (int, optional): Decodes audio file to this sample rate. Defaults to fingerprint.DEFAULT_FS.
            plot (bool, optional): Plots. Defaults to False.
            max_lags (float, optional): Maximum lags in seconds.
            kwargs: additional arguments for scipy.signal.find_peaks.

        Returns:
            dict: dictionary of recognition information
        """
        return correcognize.correcognize(
            target_file_path,
            against_file_path,
            start_end_target=start_end_target,
            start_end_against=start_end_against,
            filter_matches=filter_matches,
            match_len_filter=match_len_filter,
            sample_rate=sample_rate,
            plot=plot,
            max_lags=max_lags,
            **kwargs,
        )

    def visrecognize(
        self,
        target_file_path: str,
        against_file_path: str,
        start_end_target: tuple = None,
        start_end_against: tuple = None,
        img_width: float = 1.0,
        volume_threshold: float = 215.0,
        volume_floor: float = 10.0,
        vert_scaling: float = 1.0,
        horiz_scaling: float = 1.0,
        calc_mse: bool = False,
        plot: bool = False,
    ) -> dict:
        """Recognize target file against against file visually.
        Uses image processing similarity techniques to identify areas with similar spectrums.
        Uses multiprocessing if multiprocessing variable is set to true
        Uses audalign freq_threshold as well

        Args:
            target_file_path (str): File to recognize.
            against_file_path (str): Recognize against.
            start_end_target (tuple(float, float), optional): Silences before and after start and end. (0, -1) Silences last second, (5.4, 0) silences first 5.4 seconds
            start_end_against (tuple(float, float), optional): Silences before and after start and end. (0, -1) Silences last second, (5.4, 0) silences first 5.4 seconds
            img_width (float): width of spectrogram image for recognition.
            volume_threshold (float): doesn't find stats for sections with max volume below threshold.
            volume_floor (float): ignores volume levels below floow.
            vert_scaling (float): scales vertically to speed up calculations. Smaller numbers have smaller images.
            horiz_scaling (float): scales horizontally to speed up calculations. Smaller numbers have smaller images. Affects alignment granularity.
            calc_mse (bool): also calculates mse for each shift if true. If false, uses default mse 20000000
            plot (bool): plot the spectrogram of each audio file.

        Returns
        -------
        match_result : dict
            dictionary containing match time and match info

            or

            None : if no match
        """
        return visrecognize.visrecognize(
            target_file_path,
            against_file_path,
            start_end_target=start_end_target,
            start_end_against=start_end_against,
            img_width=img_width,
            volume_threshold=volume_threshold,
            volume_floor=volume_floor,
            vert_scaling=vert_scaling,
            horiz_scaling=horiz_scaling,
            calc_mse=calc_mse,
            use_multiprocessing=self.multiprocessing,
            num_processes=self.num_processors,
            plot=plot,
        )

    def correcognize_directory(
        self,
        target_file_path: str,
        against_directory: str,
        start_end: tuple = None,
        filter_matches: float = 0.5,
        match_len_filter: int = None,
        sample_rate: int = fingerprint.DEFAULT_FS,
        max_lags: float = None,
        plot: bool = False,
        _file_audsegs: dict = None,
        **kwargs,
    ):
        """Uses cross correlation to find alignment

        Faster than visrecognize or recognize and more useful for amplitude
        based alignments

        Args:
            target_file_path (str): File to recognize
            against_directory (str): Directory to recognize against
            start_end (tuple(float, float), optional): Silences before and after start and end. (0, -1) Silences last second, (5.4, 0) silences first 5.4 seconds
            filter_matches (float, optional): Filters based on confidence. Ranges between 0 and 1. Defaults to 0.5.
            match_len_filter (int, optional): Limits number of matches returned. Defaults to 30.
            sample_rate (int, optional): Decodes audio file to this sample rate. Defaults to fingerprint.DEFAULT_FS.
            plot (bool, optional): Plots. Defaults to False.
            max_lags (float, optional): Maximum lags in seconds.
            _file_audsegs (dict, optional): For use with align.
            kwargs: additional arguments for scipy.signal.find_peaks.

        Returns:
            dict: dictionary of recognition information
        """
        return correcognize.correcognize_directory(
            target_file_path,
            against_directory,
            start_end=start_end,
            filter_matches=filter_matches,
            match_len_filter=match_len_filter,
            sample_rate=sample_rate,
            plot=plot,
            max_lags=max_lags,
            _file_audsegs=_file_audsegs,
            **kwargs,
        )

    def visrecognize_directory(
        self,
        target_file_path: str,
        against_directory: str,
        start_end: tuple = None,
        img_width: float = 1.0,
        volume_threshold: float = 215.0,
        volume_floor: float = 10.0,
        vert_scaling: float = 1.0,
        horiz_scaling: float = 1.0,
        calc_mse: bool = False,
        plot: bool = False,
    ) -> dict:
        """Recognize target file against against directory visually.
        Uses image processing similarity techniques to identify areas with similar spectrums.
        Uses multiprocessing if multiprocessing variable is set to true
        Uses audalign freq_threshold as well

        Args:
            target_file_path (str): File to recognize.
            against_directory (str): Recognize against all files in directory.
            start_end (tuple(float, float), optional): Silences before and after start and end. (0, -1) Silences last second, (5.4, 0) silences first 5.4 seconds
            img_width (float): width of spectrogram image for recognition.
            volume_threshold (int): doesn't find stats for sections with max volume below threshold.
            volume_floor (float): ignores volume levels below floow.
            vert_scaling (float): scales vertically to speed up calculations. Smaller numbers have smaller images.
            horiz_scaling (float): scales horizontally to speed up calculations. Smaller numbers have smaller images. Affects alignment granularity.
            calc_mse (bool): also calculates mse for each shift if true. If false, uses default mse 20000000
            plot (bool): plot the spectrogram of each audio file.

        Returns
        -------
        match_result : dict
            dictionary containing match time and match info

            or

            None : if no match
        """
        return visrecognize.visrecognize_directory(
            target_file_path,
            against_directory,
            start_end=start_end,
            img_width=img_width,
            volume_threshold=volume_threshold,
            volume_floor=volume_floor,
            vert_scaling=vert_scaling,
            horiz_scaling=horiz_scaling,
            calc_mse=calc_mse,
            use_multiprocessing=self.multiprocessing,
            num_processes=self.num_processors,
            plot=plot,
        )

    def write_processed_file(
        self,
        file_path: str,
        destination_file: str,
        start_end: tuple = None,
    ) -> None:
        """
        writes given file to the destination file after processing for fingerprinting

        Args
            file_path (str): file path of audio file
            destination_file (str): file path and name to write file to
            start_end (tuple(float, float), optional): Silences before and after start and end. (0, -1) Silences last second, (5.4, 0) silences first 5.4 seconds

        Returns
        -------
        None
        """
        filehandler.read(
            filename=file_path,
            wrdestination=destination_file,
            start_end=start_end,
        )

    def plot(
        self,
        file_path: str,
        start_end: tuple = None,
    ) -> None:
        """
        Plots the file_path's peak chart

        Args
            file_path (str): file to plot
            start_end (tuple(float, float), optional): Silences before and after start and end. (0, -1) Silences last second, (5.4, 0) silences first 5.4 seconds

        Returns
        -------
        None
        """
        self._fingerprint_file(file_path, start_end=start_end, plot=True)

    def clear_fingerprints(self) -> None:
        """
        Resets audalign object to brand new state

        Args
            None

        Returns
        -------
        None
        """
        self.file_names = []
        self.fingerprinted_files = []
        self.total_fingerprints = 0

    def target_align(
        self,
        target_file: str,
        directory_path: str,
        destination_path: str = None,
        start_end: tuple = None,
        write_extension: str = None,
        technique: str = "fingerprints",
        alternate_strength_stat: str = None,
        filter_matches: float = None,
        locality: float = None,
        locality_filter_prop: float = None,
        volume_threshold: float = 216,
        volume_floor: float = 10.0,
        vert_scaling: float = 1.0,
        horiz_scaling: float = 1.0,
        img_width: float = 1.0,
        calc_mse: bool = False,
        cor_sample_rate: int = fingerprint.DEFAULT_FS,
        **kwargs,
    ):
        """matches and relative offsets for all files in directory_path using only target file,
        aligns them, and writes them to destination_path if given. Uses fingerprinting by defualt,
        but uses visual recognition if false

        Args:
            target_file (str): File to find alignments against
            directory_path (str): Directory to align against
            destination_path (str, optional): Directory to write alignments to
            start_end (tuple(float, float), optional): Silences before and after start and end. (0, -1) Silences last second, (5.4, 0) silences first 5.4 seconds
            write_extension (str, optional): audio file format to write to. Defaults to None.
            technique (str, optional): options are "fingerprints", "visual", "correlation"
            alternate_strength_stat (str, optional): confidence for fingerprints, ssim for visual, mse or count also work for visual. Defaults to None.
            filter_matches (int, float, optional): filter matches level for fingerprinting. Defaults to 1.
            locality (float, optional): In seconds for fingerprints, only matches files within given window sizes
            locality_filter_prop (int, float,optional): within each offset, filters locality tuples by proportion of highest confidence to tuple confidence
            volume_threshold (float, optional): volume threshold for visual recognition. Defaults to 216.
            volume_floor (float): ignores volume levels below floow.
            vert_scaling (float): scales vertically to speed up calculations. Smaller numbers have smaller images.
            horiz_scaling (float): scales horizontally to speed up calculations. Smaller numbers have smaller images. Affects alignment granularity.
            img_width (float, optional): width of image comparison for visual recognition
            calc_mse (bool): also calculates mse for each shift if true. If false, uses default mse 20000000
            cor_sample_rate (int): optionally change the sample rate if using correlation
            kwargs: additional arguments for scipy.signal.find_peaks

        Returns:
            dict: dict of file name with shift as value along with match info
        """
        self.file_names, temp_file_names = [], self.file_names
        self.fingerprinted_files, temp_fingerprinted_files = (
            [],
            self.fingerprinted_files,
        )
        self.total_fingerprints, temp_total_fingerprints = 0, self.total_fingerprints

        try:
            if alternate_strength_stat == "mse":
                calc_mse = True

            target_name = os.path.basename(target_file)
            total_alignment = {}
            file_names_and_paths = {}

            if technique == "fingerprints":

                if filter_matches is None:
                    filter_matches = 1
                if locality_filter_prop is None or locality_filter_prop > 1.0:
                    locality_filter_prop = 1.0

                if start_end is not None:
                    self.fingerprint_file(target_file, start_end=start_end)

                all_against_files = filehandler.find_files(directory_path)
                all_against_files_full = [x[0] for x in all_against_files]
                all_against_files_base = [
                    os.path.basename(x) for x in all_against_files_full
                ]
                if (
                    os.path.basename(target_file) in all_against_files_base
                    and target_file not in all_against_files_full
                ):
                    self.fingerprint_file(target_file)
                self.fingerprint_directory(directory_path)

                alignment = self.recognize(
                    target_file,
                    filter_matches=filter_matches,
                    locality=locality,
                    locality_filter_prop=locality_filter_prop,
                    start_end=start_end,
                )

            elif technique == "visual":
                alignment = self.visrecognize_directory(
                    target_file_path=target_file,
                    against_directory=directory_path,
                    start_end=start_end,
                    volume_threshold=volume_threshold,
                    volume_floor=volume_floor,
                    vert_scaling=vert_scaling,
                    horiz_scaling=horiz_scaling,
                    img_width=img_width,
                    calc_mse=calc_mse,
                )
            elif technique == "correlation":
                alignment = self.correcognize_directory(
                    target_file_path=target_file,
                    against_directory=directory_path,
                    start_end=start_end,
                    filter_matches=filter_matches,
                    sample_rate=cor_sample_rate,
                    **kwargs,
                )
            else:
                raise NameError(
                    f'Technique parameter must be fingerprint, visual, or correlation, not "{technique}"'
                )

            file_names_and_paths[target_name] = target_file
            total_alignment[target_name] = alignment

            if not alignment:
                print("No results")
                return

            for file_path, _ in filehandler.find_files(directory_path):
                if (
                    os.path.basename(file_path)
                    in total_alignment[target_name]["match_info"].keys()
                ):
                    file_names_and_paths[os.path.basename(file_path)] = file_path

            if not alternate_strength_stat:
                if technique == "fingerprints":
                    alternate_strength_stat = self.CONFIDENCE
                elif technique == "visual":
                    alternate_strength_stat = "ssim"
                else:
                    alternate_strength_stat = self.CONFIDENCE
            files_shifts = align.find_most_matches(
                total_alignment, strength_stat=alternate_strength_stat
            )
            if not files_shifts:
                return

            if destination_path:
                try:
                    # Make target directory
                    if not os.path.exists(destination_path):
                        os.makedirs(destination_path)
                    self._write_shifted_files(
                        files_shifts,
                        destination_path,
                        file_names_and_paths,
                        write_extension,
                    )
                except PermissionError:
                    print("Permission Denied for write align")

            print(
                f"{len(files_shifts)} out of {len(file_names_and_paths)} found and aligned"
            )

            files_shifts["match_info"] = total_alignment
            files_shifts["names_and_paths"] = file_names_and_paths
            return files_shifts

        finally:
            self.file_names = temp_file_names
            self.fingerprinted_files = temp_fingerprinted_files
            self.total_fingerprints = temp_total_fingerprints

    def align_files(
        self,
        filename_a,
        filename_b,
        *filenames,
        destination_path: str = None,
        write_extension: str = None,
        technique: str = "fingerprints",
        filter_matches: float = None,
        locality: float = None,
        locality_filter_prop: float = None,
        cor_sample_rate: int = fingerprint.DEFAULT_FS,
        max_lags: float = None,
        **kwargs,
    ):
        """
        Finds matches and relative offsets for all files given, aligns them, and writes them to destination_path if given

        try align_files(*filepath_list) for more concision if desired

        Args
            filename_a (str): String of path for alignment
            filename_b (str): String of path for alignment
            *filenames (strs): strings of paths for alignment
            destination_path (str): String of path to write alignments to
            write_extension (str): if given, writes all alignments with given extension (ex. ".wav" or "wav")
            technique (str): either "fingerprints" or "correlation"
            filter_matches (float): filters based on confidence.
            locality (float): Only recognizes against fingerprints in given width. In seconds
            locality_filter_prop (int, float,optional): within each offset, filters locality tuples by proportion of highest confidence to tuple confidence
            cor_sample_rate (int): Sampling rate for correlation
            max_lags (float, optional): Maximum lags in seconds for correlation.
            **kwargs: Additional arguments for finding peaks in correlation

        Returns
        -------
            files_shifts (dict{float}): dict of file name with shift as value
        """
        filename_list = [filename_a, filename_b, *filenames]
        # print(filename_a, filename_b, *filenames, sep="\n")
        # print(filename_list)
        return self._align(
            filename_list=filename_list,
            file_dir=None,
            destination_path=destination_path,
            write_extension=write_extension,
            technique=technique,
            filter_matches=filter_matches,
            locality=locality,
            locality_filter_prop=locality_filter_prop,
            cor_sample_rate=cor_sample_rate,
            max_lags=max_lags,
            **kwargs,
        )

    def align(
        self,
        directory_path: str,
        destination_path: str = None,
        write_extension: str = None,
        technique: str = "fingerprints",
        filter_matches: float = None,
        locality: float = None,
        locality_filter_prop: float = None,
        cor_sample_rate: int = fingerprint.DEFAULT_FS,
        max_lags: float = None,
        **kwargs,
    ):
        """
        Finds matches and relative offsets for all files in directory_path, aligns them, and writes them to destination_path

        Args
            directory_path (str): String of directory for alignment
            destination_path (str): String of path to write alignments to
            write_extension (str): if given, writes all alignments with given extension (ex. ".wav" or "wav")
            technique (str): either "fingerprints" or "correlation"
            filter_matches (float): filters based on confidence.
            locality (float): Only recognizes against fingerprints in given width. In seconds
            locality_filter_prop (int, float,optional): within each offset, filters locality tuples by proportion of highest confidence to tuple confidence
            cor_sample_rate (int): Sampling rate for correlation
            **kwargs: Additional arguments for finding peaks in correlation

        Returns
        -------
            files_shifts (dict{float}): dict of file name with shift as value
        """
        return self._align(
            filename_list=None,
            file_dir=directory_path,
            destination_path=destination_path,
            write_extension=write_extension,
            technique=technique,
            filter_matches=filter_matches,
            locality=locality,
            locality_filter_prop=locality_filter_prop,
            cor_sample_rate=cor_sample_rate,
            max_lags=max_lags,
            **kwargs,
        )

    def _align(
        self,
        filename_list: str,
        file_dir: str,
        destination_path: str = None,
        write_extension: str = None,
        technique: str = "fingerprints",
        filter_matches: float = None,
        locality: float = None,
        locality_filter_prop: float = None,
        cor_sample_rate: int = fingerprint.DEFAULT_FS,
        max_lags: float = None,
        fine_aud_file_dict: dict = None,
        **kwargs,
    ):

        self.file_names, temp_file_names = [], self.file_names
        self.fingerprinted_files, temp_fingerprinted_files = (
            [],
            self.fingerprinted_files,
        )
        self.total_fingerprints, temp_total_fingerprints = 0, self.total_fingerprints

        try:

            # Make target directory
            if destination_path:
                if not os.path.exists(destination_path):
                    os.makedirs(destination_path)

            if technique == "fingerprints":
                if filter_matches is None:
                    filter_matches = 1
                if locality_filter_prop is None or locality_filter_prop > 1.0:
                    locality_filter_prop = 1.0
                if file_dir:
                    self.fingerprint_directory(file_dir)
                else:
                    self.fingerprint_directory(
                        filename_list,
                        _file_audsegs=fine_aud_file_dict,
                    )
            elif technique == "correlation":
                if file_dir:
                    self.file_names = filehandler.get_audio_files_directory(file_dir)
                elif fine_aud_file_dict:
                    self.file_names = [
                        os.path.basename(x) for x in fine_aud_file_dict.keys()
                    ]
                else:
                    self.file_names = [os.path.basename(x) for x in filename_list]

            else:
                raise NameError(
                    f'Technique parameter must be fingerprints, visual, or correlation, not "{technique}"'
                )

            total_alignment = {}
            file_names_and_paths = {}

            if file_dir:
                file_list = filehandler.find_files(file_dir)
                dir_or_list = file_dir
            elif fine_aud_file_dict:
                file_list = zip(
                    fine_aud_file_dict.keys(), ["_"] * len(fine_aud_file_dict)
                )
                dir_or_list = fine_aud_file_dict.keys()
            else:
                file_list = zip(filename_list, ["_"] * len(filename_list))
                dir_or_list = filename_list
            # Get matches and paths
            for file_path, _ in file_list:
                name = os.path.basename(file_path)
                if name in self.file_names:
                    if technique == "fingerprints":
                        alignment = self.recognize(
                            file_path,
                            filter_matches=filter_matches,
                            locality=locality,
                            locality_filter_prop=locality_filter_prop,
                            max_lags=max_lags,
                        )
                    elif technique == "correlation":
                        alignment = self.correcognize_directory(
                            file_path,
                            dir_or_list,
                            filter_matches=filter_matches,
                            sample_rate=cor_sample_rate,
                            _file_audsegs=fine_aud_file_dict,
                            max_lags=max_lags,
                            **kwargs,
                        )
                    file_names_and_paths[name] = file_path
                    total_alignment[name] = alignment

            files_shifts = align.find_most_matches(total_alignment)
            if not files_shifts:
                return
            files_shifts = align.find_matches_not_in_file_shifts(
                total_alignment, files_shifts
            )

            if destination_path:
                try:
                    self._write_shifted_files(
                        files_shifts,
                        destination_path,
                        file_names_and_paths,
                        write_extension,
                    )
                except PermissionError:
                    print("Permission Denied for write align")

            print(
                f"{len(files_shifts)} out of {len(file_names_and_paths)} found and aligned"
            )

            files_shifts["match_info"] = total_alignment
            files_shifts["names_and_paths"] = file_names_and_paths
            return files_shifts

        finally:
            self.file_names = temp_file_names
            self.fingerprinted_files = temp_fingerprinted_files
            self.total_fingerprints = temp_total_fingerprints

    def fine_align(
        self,
        results,
        destination_path: str = None,
        write_extension: str = None,
        technique="correlation",
        filter_matches: float = None,
        max_lags: float = 2,
        locality: float = None,
        locality_filter_prop: float = None,
        cor_sample_rate: int = fingerprint.DEFAULT_FS,
        match_index: int = 0,
        strength_stat: str = CONFIDENCE,
        **kwargs,
    ):
        """
        Finds matches and relative offsets for all files in directory_path, aligns them, and writes them to destination_path

        Args
            results (dict): results from previous alignments.
            directory_path (str): String of directory for alignment
            destination_path (str): String of path to write alignments to
            write_extension (str): if given, writes all alignments with given extension (ex. ".wav" or "wav")
            technique (str): either "fingerprints" or "correlation"
            filter_matches (float): filters based on confidence.
            locality (float): Only recognizes against fingerprints in given width. In seconds
            locality_filter_prop (int, float,optional): within each offset, filters locality tuples by proportion of highest confidence to tuple confidence
            cor_sample_rate (int): Sampling rate for correlation
            match_index (int): reorders the input results to the given match index.
            strength_stat (str): strength stat for finding proper alignments. Defaults to confidence.
            **kwargs: Additional arguments for finding peaks in correlation

        Returns
        -------
            files_shifts (dict{float}): dict of file name with shift as value. Includes match_info, fine_match_info, and names_and_paths
        """
        print("Fine Aligning...")

        if match_index != 0:
            recalc_shifts_results = align.recalc_shifts_index(
                results, strength_stat=strength_stat, match_index=match_index
            )
            paths_audio = filehandler.shift_get_files(
                recalc_shifts_results, sample_rate=cor_sample_rate
            )
        else:
            paths_audio = filehandler.shift_get_files(
                results, sample_rate=cor_sample_rate
            )

        new_results = self._align(
            filename_list=None,
            file_dir=None,
            technique=technique,
            filter_matches=filter_matches,
            locality=locality,
            locality_filter_prop=locality_filter_prop,
            cor_sample_rate=cor_sample_rate,
            max_lags=max_lags,
            fine_aud_file_dict=paths_audio,
            **kwargs,
        )
        if new_results is None:
            print("No matches found for fine alignment")
            return
        new_results = align.combine_fine(results, new_results)

        if destination_path:
            copy_dict = {}
            for name, value in new_results.items():
                if name not in ["names_and_paths", "match_info", "fine_match_info"]:
                    copy_dict[name] = value
            try:
                self._write_shifted_files(
                    copy_dict,
                    destination_path,
                    new_results["names_and_paths"],
                    write_extension,
                )
            except PermissionError:
                print("Permission Denied for write fine_align")

        return new_results

    @staticmethod
    def _write_shifted_files(
        files_shifts: dict,
        destination_path: str,
        names_and_paths: dict,
        write_extension: str,
    ):
        """
        Writes files to destination_path with specified shift

        Args
            files_shifts (dict{float}): dict with file path as key and shift as value
            destination_path (str): folder to write file to
            names_and_paths (dict{str}): dict with name as key and path as value
        """
        filehandler.shift_write_files(
            files_shifts, destination_path, names_and_paths, write_extension
        )

    @staticmethod
    def get_metadata(file_path: str):
        """Returns metadata of audio or video file

        if file_path is not a valid file or is a directory, returns empty dict

        Args:
            file_path (str): file path to file

        Returns:
            dict: dict of tags and values
        """
        return mediainfo(filepath=file_path)

    @staticmethod
    def write_shifted_file(
        file_path: str, destination_path: str, offset_seconds: float
    ):
        """
        Writes file to destination_path with specified shift in seconds

        Args
            file_path (str): file path of file to shift
            destination_path (str): where to write file to and file name
            offset_seconds (float): how many seconds to shift, can't be negative
        """
        filehandler.shift_write_file(file_path, destination_path, offset_seconds)

    @staticmethod
    def convert_audio_file(
        file_path: str,
        destination_path: str,
        start_end: tuple = None,
    ):
        """
        Convert audio file to type specified in destination path

        Args
            file_path (str): file path of file to shift
            destination_path (str): where to write file to and file name
            start_end (tuple(float, float), optional): Silences before and after start and end. (0, -1) Silences last second, (5.4, 0) silences first 5.4 seconds
        """
        filehandler.read(
            filename=file_path, wrdestination=destination_path, start_end=start_end
        )

    @staticmethod
    def remove_noise_file(
        filepath: str,
        noise_start: float,
        noise_end: float,
        destination: str,
        alt_noise_filepath: str = None,
        prop_decrease: float = 1,
        use_tensorflow: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        """Remove noise from audio file by specifying start and end seconds of representative sound sections. Writes file to destination

        Args:
            filepath (str): filepath to read audio file
            noise_start (float): positition in seconds of start of noise section
            noise_end (float): position in seconds of end of noise section
            destination (str): filepath of destination to write to
            alt_noise_filepath (str): path of different file for noise sample
            prop_decrease (float): between 0 and 1. Proportion to decrease noise
            use_tensorflow (bool, optional): Uses tensorflow to increase speed if available. Defaults to False.
            verbose (bool, optional): Shows several plots of noise removal process. Defaults to False.
            kwargs : kwargs for noise reduce. Look at noisereduce kwargs
        """
        filehandler.noise_remove(
            filepath,
            noise_start,
            noise_end,
            destination,
            alt_noise_filepath=alt_noise_filepath,
            prop_decrease=prop_decrease,
            use_tensorflow=use_tensorflow,
            verbose=verbose,
            **kwargs,
        )

    def remove_noise_directory(
        self,
        directory: str,
        noise_filepath: str,
        noise_start: float,
        noise_end: float,
        destination_directory: str,
        prop_decrease: float = 1,
        use_tensorflow: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        """Remove noise from audio files in directory by specifying start and end seconds of representative sound sections. Writes file to destination directory
        Uses multiprocessing if self.multiprocessing is true

        Args:
            directory (str): filepath for directory to quiet
            nosie_filepath (str): filepath to read noise file
            noise_start (float): positition in seconds of start of noise section
            noise_end (float): position in seconds of end of noise section
            destination_directory (str): filepath of destination directory to write to
            prop_decrease (float): between 0 and 1. Proportion to decrease noise
            use_tensorflow (bool, optional): Uses tensorflow to increase speed if available. Defaults to False.
            verbose (bool, optional): Shows several plots of noise removal process. Defaults to False.
            kwargs : kwargs for noise reduce. Look at noisereduce kwargs
        """
        filehandler.noise_remove_directory(
            directory,
            noise_filepath,
            noise_start,
            noise_end,
            destination_directory,
            prop_decrease=prop_decrease,
            use_tensorflow=use_tensorflow,
            verbose=verbose,
            use_multiprocessing=self.multiprocessing,
            num_processes=self.num_processors,
            **kwargs,
        )
