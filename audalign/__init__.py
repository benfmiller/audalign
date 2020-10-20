import audalign.filehandler as filehandler
import audalign.fingerprint as fingerprint
import audalign.recognize as recognize
import audalign.align as align
from functools import partial
import multiprocessing
import os
import traceback
import sys
import pickle
import json


class Audalign:

    # Names that appear in match information
    CONFIDENCE = "confidence"
    MATCH_TIME = "match_time"
    OFFSET_SAMPLES = "offset_samples"
    OFFSET_SECS = "offset_seconds"

    def __init__(
        self,
        *args,
        multiprocessing=True,
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

        Parameters
        ----------
        arg1 : str
            Optional file path to load json or pickle file of already fingerprinted files
        multiprocessing : bool
            option to turn off multiprocessing
        hash_style : str
            which hash style to use : ['base','panako_mod','panako', 'base_three']
        accuracy : int
            which accuracy level 1-4
        threshold: int
            filters fingerprints below threshold
        noisereduce: bool
            runs noise reduce on audio
        """

        self.set_freq_threshold(freq_threshold)

        self.file_names = []
        self.fingerprinted_files = []
        self.multiprocessing = multiprocessing
        self.total_fingerprints = 0

        if len(args) > 0:
            self.load_fingerprinted_files(args[0])

        self.hash_style = hash_style

        if accuracy != 2:
            self.set_accuracy(accuracy)

    def set_accuracy(self, accuracy):
        """
        Sets the accuracy level of audalign object

        There are four accuracy levels with 1 being the lowest accuracy but the fastest. 3 is the highest recommended.
        4 gives the highest accuracy, but can take several gigabytes of memory for a couple files.
        Accuracy settings are acheived by manipulations in fingerprinting variables.
        
        Specific values for accuracy levels were chosen semi-arbitrarily from experimentation to give a few good options.

        Parameters
        ----------
            accuracy : int
                which accuracy level: 1-4
        """
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

    def set_freq_threshold(self, threshold):
        """[Sets minimum frequency threshold for fingerprint]

        Args:
            threshold ([int]): [threshold]
        """
        fingerprint.threshold = threshold

    def save_fingerprinted_files(self, filename: str) -> None:
        """
        Serializes fingerprinted files to json or pickle file

        Parameters
        ----------
        filename
            must be either json or pickle extension

        Returns
        -------
        None
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

        Parameters
        ----------
        filename : str
            must be either json or pickle extension

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

    def filter_duplicates(self):
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
        self, path: str, plot=False, nprocesses=None, extensions=["*"]
    ) -> None:
        """
        Fingerprints all files in given directory and all subdirectories

        Parameters
        ----------
        path : str
            path to directory to be fingerprinted
        plot : boolean
            if true, plots the peaks to be fingerprinted on a spectrogram
        nprocesses : int
            specifies number of threads to use
        extensions : list[str]
            specify which extensions to fingerprint

        Returns
        -------
        None
        """

        result = self._fingerprint_directory(path, plot, nprocesses, extensions)

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
        self, path, plot=False, nprocesses=None, extensions=["*"]
    ):
        """
        Worker function for fingerprint_directory

        Fingerprints all files in given directory and all subdirectories

        Parameters
        ----------
        path : str
            path to directory to be fingerprinted
        plot : boolean
            if true, plots the peaks to be fingerprinted on a spectrogram
        nprocesses : int
            specifies number of threads to use
        extensions : list[str]
            specify which extensions to fingerprint

        Returns
        -------
        None
        """

        filenames_to_fingerprint = []
        for filename, _ in filehandler.find_files(
            path, extensions
        ):  # finds all files to fingerprint
            file_name = os.path.basename(filename)
            if file_name in self.file_names:
                print(f"{file_name} already fingerprinted")
                continue
            filenames_to_fingerprint.append(filename)

        if len(filenames_to_fingerprint) == 0:
            print("Directory contains 0 files or could not be found")
            return

        _fingerprint_worker_directory = partial(
            _fingerprint_worker, hash_style=self.hash_style, plot=plot,
        )

        if self.multiprocessing == True:

            # Try to use the maximum amount of processes if not given.
            try:
                nprocesses = nprocesses or multiprocessing.cpu_count()
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
                try:
                    file_name = os.path.basename(filename)
                    if file_name in self.file_names:
                        print(f"{file_name} already fingerprinted, continuing...")
                        continue
                    file_name, hashes = _fingerprint_worker_directory(filename)
                    result.append([file_name, hashes])
                except Exception:
                    print(f'Failed fingerprinting "{filename}"')
                    # Print traceback because we can't reraise it here
                    traceback.print_exc(file=sys.stdout)
        return result

    def fingerprint_file(self, file_path, set_file_name=None, plot=False):
        """
        Fingerprints given file and adds to fingerprinted files

        Parameters
        ----------
        file_path : str
            path to word to be fingerprinted
        set_file_name : str
            option to set file name manually rather than use file name in file_path
        plot : boolean
            if true, plots the peaks to be fingerprinted on a spectrogram

        Returns
        -------
        None
        """
        file_name = os.path.basename(file_path)
        if file_name in self.file_names:
            print(f"{file_name} already fingerprinted")
            return None

        file_name, hashes = self._fingerprint_file(file_path, set_file_name, plot)
        if file_name != None:
            self.fingerprinted_files.append([file_name, hashes])
            self.file_names.append(file_name)
            self.total_fingerprints += len(hashes)

    def _fingerprint_file(self, file_path, set_file_name=None, plot=False):
        """
        Worker function for fingerprint_file

        Fingerprints given file and adds to fingerprinted files

        Parameters
        ----------
        file_path : str
            path to word to be fingerprinted
        set_file_name : str
            option to set file name manually rather than use file name in file_path
        plot : boolean
            if true, plots the peaks to be fingerprinted on a spectrogram

        Returns
        -------
        [file_name, hashes]
        """

        file_name, hashes = _fingerprint_worker(file_path, self.hash_style, plot=plot,)
        file_name = set_file_name or file_name
        return [file_name, hashes]

    def recognize(self, file_path, filter_matches=1, *args, **kwargs):
        """
        Recognizes given file against already fingerprinted files

        Offset describes duration that the recognized file aligns after the target file
        Does not recognize against files with same name and extention

        Parameters
        ----------
        file_path : str
            file path of target file to recognize
        filter_matches : int
            filters all matches lower than given argument, 1 is recommended

        Returns
        -------
        match_result : dict
            dictionary containing match time and match info

            or

            None : if no match
        """
        return recognize.recognize(self, file_path, filter_matches, *args, **kwargs)

    def write_processed_file(self, file_path, destination_file):
        """
        writes given file to the destination file after processing for fingerprinting

        Parameters
        ----------
        file_path : str
            file path of audio file
        destination_file : str
            file path and name to write file to

        Returns
        -------
        None
        """
        filehandler.read(file_path, wrdestination=destination_file)

    def plot(self, file_path):
        """
        Plots the file_path's peak chart

        Parameters
        ----------
        file_path : str
            file to plot

        Returns
        -------
        None
        """
        self._fingerprint_file(file_path, plot=True)

    def clear_fingerprints(self):
        """
        Resets audalign object to brand new state

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.file_names = []
        self.fingerprinted_files = []
        self.total_fingerprints = 0

    def align(
        self,
        directory_path,
        destination_path=None,
        write_extension=None,
        filter_matches=1,
    ):
        """
        Finds matches and relative offsets for all files in directory_path, aligns them, and writes them to destination_path

        Parameters
        ----------
        directory_path : str
            String of directory for alignment

        destination_path : str
            String of path to write alignments to

        write_extension : str
            if given, writes all alignments with given extension (ex. ".wav" or "wav")

        Returns
        -------
        files_shifts : dict{float}
            dict of file name with shift as value
        """

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

            self.fingerprint_directory(directory_path)

            total_alignment = {}
            file_names_and_paths = {}

            # Get matches and paths
            for file_path, _ in filehandler.find_files(directory_path):
                name = os.path.basename(file_path)
                if name in self.file_names:
                    alignment = self.recognize(file_path, filter_matches=filter_matches)
                    file_names_and_paths[name] = file_path
                    total_alignment[name] = alignment

            files_shifts = align.find_most_matches(total_alignment)
            if not files_shifts:
                return
            files_shifts = align.find_matches_not_in_file_shifts(
                total_alignment, files_shifts
            )

            if destination_path:
                self._write_shifted_files(
                    files_shifts,
                    destination_path,
                    file_names_and_paths,
                    write_extension,
                )

            print(
                f"{len(files_shifts)} out of {len(file_names_and_paths)} found and aligned"
            )

            files_shifts["match_info"] = total_alignment
            return files_shifts

        finally:
            self.file_names = temp_file_names
            self.fingerprinted_files = temp_fingerprinted_files
            self.total_fingerprints = temp_total_fingerprints

    def _write_shifted_files(
        self, files_shifts, destination_path, names_and_paths, write_extension
    ):
        """
        Writes files to destination_path with specified shift

        Parameters
        ----------
        files_shifts : dict{float}
            dict with file path as key and shift as value
        destination_path : str
            folder to write file to
        names_and_paths : dict{str}
            dict with name as key and path as value
        """
        filehandler.shift_write_files(
            files_shifts, destination_path, names_and_paths, write_extension
        )

    def write_shifted_file(self, file_path, destination_path, offset_seconds):
        """
        Writes file to destination_path with specified shift in seconds

        Parameters
        ----------
        file_path : str
            file path of file to shift
        destination_path : str
            where to write file to and file name
        offset_seconds : float
            how many seconds to shift, can't be negative
        """
        filehandler.shift_write_file(file_path, destination_path, offset_seconds)

    def convert_audio_file(self, file_path, destination_path):
        """
        Convert audio file to type specified in destination path

        Parameters
        ----------
        file_path : str
            file path of file to shift
        destination_path : str
            where to write file to and file name
        """
        filehandler.convert_audio_file(file_path, destination_path)

    def remove_noise_file(self, filepath, noise_start, noise_end, destination, use_tensorflow=False, verbose=False):
        filehandler.noise_remove(filepath, noise_start, noise_end, destination, use_tensorflow=use_tensorflow, verbose=verbose)

    def remove_noise_directory(self, directory, noise_filepath, noise_start, noise_end, destination_directory):
        # getting file going first
        pass


def _fingerprint_worker(file_path: str, hash_style="panako_mod", plot=False,) -> None:
    """
    Runs the file through the fingerprinter and returns file_name and hashes

    Parameters
    ----------
    file_path : str
        file_path to be fingerprinted
    plot : bool
        displays the plot of the peaks if true
    amp_min : int
        minimum amplitude to be considered a peak

    Returns
    -------
    file_name : str, hashes : dict{str: [int]}
        file_name and hash dictionary
    """

    file_name = os.path.basename(file_path)

    try:
        channel, _ = filehandler.read(file_path)
    except FileNotFoundError:
        print(f'"{file_path}" not found')
        return None, None
    except Exception:
        print(f'File "{file_name}" could not be decoded')
        return None, None

    print(f"Fingerprinting {file_name}")
    hashes = fingerprint.fingerprint(channel, hash_style=hash_style, plot=plot,)

    print(f"Finished fingerprinting {file_name}")

    return file_name, hashes
