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
import typing


class FingerprintRecognizer(BaseRecognizer):
    config: FingerprintConfig
    file_names = []
    fingerprinted_files = []
    total_fingerprints = 0
    temp_fingerprints_list = []

    def __init__(
        self, config: FingerprintConfig = None, load_fingerprints_file: str = None
    ):
        self.config = FingerprintConfig() if config is None else config
        self.file_names = []
        self.fingerprinted_files = []
        self.total_fingerprints = 0
        self.temp_fingerprints_list = []

        if load_fingerprints_file is not None:
            self.load_fingerprinted_files(load_fingerprints_file)

    def align_stat_print(self):
        print()
        print(f"Total fingerprints: {self.total_fingerprints}")

    def align_setup_files(
        self,
        filename_list: typing.Union[str, list],
        file_dir: typing.Optional[str],
        target_aligning: bool,
        fine_aud_file_dict: typing.Optional[dict],
    ):
        if target_aligning:
            if self.config.target_start_end is not None:
                self.fingerprint_file(
                    filename_list[0],
                )
            file_dir = self.prelim_fingerprint_checks(
                target_file=filename_list[0],
                directory_path=file_dir,
            )
        if file_dir:
            self.fingerprint_directory(file_dir)
        else:
            self.fingerprint_directory(
                filename_list,
                _file_audsegs=fine_aud_file_dict,
            )

    def prelim_fingerprint_checks(self, target_file, directory_path):
        all_against_files = filehandler.find_files(directory_path)
        all_against_files_full = [x[0] for x in all_against_files]
        all_against_files_base = [os.path.basename(x) for x in all_against_files_full]
        if (
            os.path.basename(target_file) in all_against_files_base
            and target_file not in all_against_files_full
        ):
            for i, x in enumerate(all_against_files_full):
                if os.path.basename(target_file) == os.path.basename(x):
                    all_against_files_full.pop(i)
                    break
            all_against_files_full.append(target_file)
        elif os.path.basename(target_file) not in all_against_files_base:
            all_against_files_full.append(target_file)
        return all_against_files_full

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

    def align_get_file_names(
        self,
        file_list: typing.Union[str, list],
        file_dir: typing.Optional[str],
        target_aligning: bool,
        fine_aud_file_dict: typing.Optional[dict],
    ) -> list:
        if target_aligning or file_dir:
            file_names = filehandler.get_audio_files_directory(file_dir, full_path=True)
        elif fine_aud_file_dict:
            file_names = fine_aud_file_dict.keys()
            for name, fingerprints in zip(self.file_names, self.fingerprinted_files):
                self.temp_fingerprints_list.append([name, fingerprints])
                self.clear_fingerprints()
        else:
            file_names = file_list

        self.fingerprint_directory(file_names, fine_aud_file_dict)

        if not fine_aud_file_dict:
            set_file_name_list = set([os.path.basename(x) for x in file_names])
            self.temp_fingerprints_list = []
            for name in list(self.file_names):
                if name not in set_file_name_list:
                    self.temp_fingerprints_list.append(self.pop_filename(name))

        file_name_list = super().align_get_file_names(
            file_list, file_dir, target_aligning, fine_aud_file_dict
        )
        return file_name_list

    def align_post_hook(
        self,
        file_list,
        dir_or_list,
        target_aligning: bool,
        fine_aud_file_dict: typing.Optional[dict],
    ):
        if fine_aud_file_dict:

            self.clear_fingerprints()
            [
                self.add_filename(name, fingerprints)
                for name, fingerprints in self.temp_fingerprints_list
            ]
        else:
            for name, fingerprints in self.temp_fingerprints_list:
                self.add_filename(name, fingerprints)
        self.temp_fingerprints_list = []

    def recognize(
        self,
        file_path: str,
        against_path: str = None,
    ) -> None:
        """
        Recognizes given file against already fingerprinted files. Will fingerprint against_path if not already fingerprinted.

        Offset describes duration that the recognized file aligns after the target file.
        Does not recognize against files with same name and extension.
        Locality option used to only return match results within certain second range, doesn't require new fingerprints.

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
        if against_path is not None:
            if os.path.isdir(against_path):
                for path in filehandler.get_audio_files_directory(
                    against_path, full_path=True
                ):
                    if path not in self.file_names and path not in to_fingerprint:
                        to_fingerprint += [path]
            elif os.path.isfile(against_path):
                if filehandler.check_is_audio_file(against_path):
                    to_fingerprint += [against_path]
        if len(to_fingerprint) > 0:
            self.fingerprint_directory(to_fingerprint)

        recognition = recognize.recognize(
            self,
            file_path=file_path,
            config=self.config,
        )

        return recognition

    def _align(self, file_path, dir_or_list):
        recognition = recognize.recognize(
            self,
            file_path=file_path,
            config=self.config,
        )
        return recognition

    def fingerprint_directory(self, path: str, _file_audsegs: dict = None) -> None:
        """
        Fingerprints all files in given directory and all subdirectories

        Args
        ----
            path (str): path to directory to be fingerprinted
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

        Returns
        -------
        None
        """
        if type(path) == str:
            file_names = filehandler.find_files(path, self.config.extensions)
        elif type(path) == list:
            file_names = zip(path, ["_"] * len(path))
            # elif path is None and _file_audsegs is not None:
        elif _file_audsegs is not None:
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
            set_file_name (str): option to set file name manually rather than use file name in file_path

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
            set_file_name (str): option to set file name manually rather than use file name in file_path

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

    def pop_filename(self, filename: str):
        """Pops filename from already fingerprinted files

        Args:
            filename (str): filename to pop

        Raises:
            KeyError: if filename is not fingerprinted

        Returns:
            [tup]: filename, fingerprinted_files entry
        """
        i = 0
        while i < len(self.file_names):
            if self.file_names[i] == filename:
                self.total_fingerprints -= len(self.fingerprinted_files[i][1])
                return self.file_names.pop(i), self.fingerprinted_files.pop(i)
            i += 1
        raise KeyError

    def add_filename(self, filename: str, file_fingerprints: list):
        if filename not in self.file_names:
            self.file_names.append(filename)
            self.fingerprinted_files.append(file_fingerprints)
            self.total_fingerprints += len(file_fingerprints[1])

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
