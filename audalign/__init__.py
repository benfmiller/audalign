import audalign.decoder as decoder
import audalign.fingerprint as fingerprint
import audalign.recognize as recognize
from functools import partial
import multiprocessing
import os
import traceback
import sys
import pickle
import json


class Audalign():

    # Names that appear in match information
    FILE_NAME = "file_name"
    CONFIDENCE = "confidence"
    MATCH_TIME = "match_time"
    OFFSET_SAMPLES = "offset_samples"
    OFFSET_SECS = "offset_seconds"

    def __init__(self, *args, multiprocessing=True):
        """
        Constructs new audalign object

        Parameters
        ----------
        arg1 : str
        Optional file path to load json or pickle file of already fingerprinted files
        multiprocessing is set to True by default
        """

        self.limit = None
        self.file_names = []
        self.fingerprinted_files = []
        self.multiprocessing = multiprocessing
        self.total_fingerprints = 0

        if len(args) > 0:
            self.load_fingerprinted_files(args[0])

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
            self.clean_fingerprinted_files()
        except FileNotFoundError:
            print(f'"{filename}" not found')

    def clean_fingerprinted_files(self):
        # TODO: clean doubles
        pass

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

        result = self.__fingerprint_directory(path, plot, nprocesses, extensions)

        for processed_file in result:
            if processed_file[0] != None:
                if processed_file[0] not in self.file_names:
                    self.fingerprinted_files.append(processed_file)
                    self.file_names.append(processed_file[0])
                    self.total_fingerprints += len(processed_file[1])

    def __fingerprint_directory(
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
        for filename, _ in decoder.find_files(
            path, extensions
        ):  # finds all files to fingerprint
            file_name = os.path.splitext(filename)
            if file_name in self.file_names:
                print(f"{file_name} already fingerprinted")
                continue
            filenames_to_fingerprint.append(filename)

        if len(filenames_to_fingerprint) == 0:
            print("Directory contains 0 files or could not be found")
            return

        _fingerprint_worker_directory = partial(
            _fingerprint_worker, limit=self.limit, plot=plot
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
                    file_name = os.path.splitext(filename)
                    if file_name in self.file_names:
                        print(f"{file_name} already fingerprinted, continuing...")
                        continue
                    file_name, hashes = _fingerprint_worker_directory(filename)
                    result.append([file_name, hashes])
                except:
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

        file_name, hashes = self.__fingerprint_file(file_path, set_file_name, plot)
        if file_name != None:
            self.fingerprinted_files.append([file_name, hashes])
            self.file_names.append(file_name)
            self.total_fingerprints += len(hashes)

    def __fingerprint_file(self, file_path, set_file_name=None, plot=False):
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
        None
        """

        file_name, extension = os.path.splitext(os.path.basename(file_path))
        file_name += extension
        if file_name in self.file_names:
            print(f"{file_name} already fingerprinted")
            return

        file_name, hashes = _fingerprint_worker(file_path, limit=self.limit, plot=plot)
        filename = decoder.path_to_filename(file_path)
        file_name = set_file_name or filename
        return [file_name, hashes]

    def recognize(self, file_path, filter_matches=1, *args, **kwargs):
        """
        Recognizes given file against already fingerprinted files

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

        if "recognizer" not in kwargs.keys():
            r = recognize.FileRecognizer(self)
        elif kwargs["recognizer"].lower() == "filerecognizer":
            r = recognize.FileRecognizer(self)
            kwargs.pop("recognizer")
        return r.recognize(file_path, filter_matches, *args, **kwargs)

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
        decoder.read(file_path, wrdestination=destination_file)

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
        _fingerprint_worker(file_path, plot=True)

    def align(self, directory_path, destination_path):

        self.file_names, temp_file_names = [], self.file_names
        self.fingerprinted_files, temp_fingerprinted_files = [], self.fingerprinted_files
        self.total_fingerprints, temp_total_fingerprints = 0, self.total_fingerprints

        try:

            if not os.path.exists(destination_path):
                os.makedirs(destination_path)
            
            self.fingerprint_directory(directory_path)

            total_alignment = {}

            for file_name, _ in decoder.find_files(directory_path):
                alignment = self.recognize()
                total_alignment
    
        finally:
            self.file_names = temp_file_names
            self.fingerprinted_files = temp_fingerprinted_files
            self.total_fingerprints = temp_total_fingerprints






def _fingerprint_worker(file_path: str, limit=None, plot=False) -> None:
    """
    Runs the file through the fingerprinter and returns file_name and hashes

    Parameters
    ----------
    file_path : str
        file_path to be fingerprinted
    limit : int
        limits fingerprinting to number of seconds from the start
    plot : bool
        displays the plot of the peaks if true
    
    Returns
    -------
    file_name : str, hashes : dict{str: [int]}
        file_name and hash dictionary
    """


    file_name, extension = os.path.splitext(os.path.basename(file_path))
    file_name += extension

    try:
        channel, Fs = decoder.read(file_path, limit)
    except FileNotFoundError:
        print(f'"{file_path}" not found')
        return None, None
    except:
        print(f'File "{file_name}" could not be decoded')
        return None, None

    print(f"Fingerprinting {file_name}")
    hashes = fingerprint.fingerprint(channel, Fs=Fs, plot=plot)
    print(f"Finished fingerprinting {file_name}")

    return file_name, hashes
