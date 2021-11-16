from audalign.recognizers import BaseRecognizer
from audalign.config.fingerprint import FingerprintConfig
import audalign.filehandler as filehandler
import audalign.recognizers.fingerprint.recognize as recognize
import audalign.recognizers.fingerprint.fingerprinter as fingerprinter

import os
import multiprocessing
from functools import partial
import pickle
import json

# ----------------------------------------------------------------------------------


# ------------------------------------------------------------------------------


class FingerprintRecognizer(BaseRecognizer):
    file_names = []
    fingerprinted_files = []
    total_fingerprints = 0

    def __init__(self, config: FingerprintConfig = None):
        # super().__init__(config=config)
        self.config = FingerprintConfig() if config is None else config

    def clear_fingerprints(self) -> None:
        """
        Resets audalign object to brand new state

        Args
        ----
            None

        Returns
        -------
        None
        """
        self.file_names = []
        self.fingerprinted_files = []
        self.total_fingerprints = 0

    def recognize(
        self,
        file_path: str,
        against_path: str,
        *args,
        **kwargs,
    ) -> None:
        """
        Recognizes given file against already fingerprinted files

        Offset describes duration that the recognized file aligns after the target file
        Does not recognize against files with same name and extension
        Locality option used to only return match results within certain second range

        Args
        ----
            file_path (str): file path of target file to recognize.
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
        if os.path.isdir(file_path):
            raise ValueError(f"file_path {file_path} must be a file")
        to_fingerprint = []
        if os.path.basename(file_path) not in self.file_names:
            to_fingerprint += [file_path]
        if os.path.isdir(against_path):
            for path in filehandler.get_audio_files_directory(
                against_path, full_path=True
            ):
                if path not in self.file_names and path not in to_fingerprint:
                    to_fingerprint += [path]
        elif os.path.isfile(against_path):
            if filehandler.check_is_audio_file(against_path):
                to_fingerprint += [against_path]
        self.fingerprint_directory(to_fingerprint)

        return recognize.recognize(
            self,
            file_path=file_path,
            config=self.config,
            *args,
            **kwargs,
        )

    def fingerprint_directory(self, path: str, _file_audsegs: dict = None) -> None:
        """
        Fingerprints all files in given directory and all subdirectories

        Args
        ----
            path (str): path to directory to be fingerprinted
            plot (boolean): if true, plots the peaks to be fingerprinted on a spectrogram
            extensions (list[str]): specify which extensions to fingerprint
            _file_audsegs (dict, None): For internal use with fine align

        Returns
        -------
        None
        """

        result = self._fingerprint_directory(path, _file_audsegs=_file_audsegs)

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
        _file_audsegs: dict = None,
    ):
        """
        Worker function for fingerprint_directory

        Fingerprints all files in given directory and all subdirectories

        Args
        ----
            path (str): path to directory to be fingerprinted
            plot (boolean): if true, plots the peaks to be fingerprinted on a spectrogram
            extensions (list[str]): specify which extensions to fingerprint

        Returns
        -------
        None
        """
        if type(path) == str:
            file_names = filehandler.find_files(path, self.config.extensions)
        elif type(path) == list:
            file_names = zip(path, ["_"] * len(path))
        elif path is None and _file_audsegs is not None:
            file_names = zip(_file_audsegs.keys(), ["_"] * len(_file_audsegs))

        one_file_already_fingerprinted = False
        filenames_to_fingerprint = []
        for filename, _ in file_names:  # finds all files to fingerprint
            file_name = os.path.basename(filename)
            if file_name in self.file_names:
                print(f"{file_name} already fingerprinted")
                one_file_already_fingerprinted = True
                continue
            filenames_to_fingerprint.append(filename)

        if len(filenames_to_fingerprint) == 0:
            if one_file_already_fingerprinted == True:
                print("All files in directory already fingerprinted")
            else:
                print("Directory contains 0 files or could not be found")
            return

        if _file_audsegs is not None:
            filenames_to_fingerprint = [
                (filename, _file_audsegs[filename])
                for filename in filenames_to_fingerprint
            ]

        _fingerprint_worker_directory = partial(
            fingerprinter._fingerprint_worker,
            config=self.config,
        )

        if self.config.multiprocessing == True:

            # Try to use the maximum amount of processes if not given.
            try:
                nprocesses = self.config.num_processors or multiprocessing.cpu_count()
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
        set_file_name: str = None,
    ) -> None:
        """
        Fingerprints given file and adds to fingerprinted files

        Args
        ----
            file_path (str): path of file to be fingerprinted
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
            file_path,
            set_file_name=set_file_name,
        )
        if file_name is not None and hashes is not None:
            self.fingerprinted_files.append([file_name, hashes])
            self.file_names.append(file_name)
            self.total_fingerprints += len(hashes)

    def _fingerprint_file(
        self,
        file_path: str,
        set_file_name: str = None,
    ):
        """
        Worker function for fingerprint_file

        Fingerprints given file and adds to fingerprinted files

        Args
        ----
            file_path (str): path to file to be fingerprinted
            start_end (tuple(float, float), optional): Silences before and after start and end. (0, -1) Silences last second, (5.4, 0) silences first 5.4 seconds
            set_file_name (str): option to set file name manually rather than use file name in file_path
            plot (boolean): if true, plots the peaks to be fingerprinted on a spectrogram

        Returns
        -------
        [file_name, hashes]
        """

        file_name, hashes = fingerprinter._fingerprint_worker(
            file_path,
            config=self.config,
        )
        file_name = set_file_name or file_name
        return [file_name, hashes]

    def save_fingerprinted_files(self, filename: str) -> None:
        """
        Serializes fingerprinted files to json or pickle file

        Args
        ----
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
        ----
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
