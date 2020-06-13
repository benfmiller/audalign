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


class Audalign(object):

    FILE_ID = "file_id"
    FILE_NAME = "file_name"
    CONFIDENCE = "confidence"
    MATCH_TIME = "match_time"
    OFFSET_SAMPLES = "offset_samples"
    OFFSET_SECS = "offset_seconds"

    def __init__(self, *args, multiprocessing=True):  # , config):

        self.limit = None
        self.file_unique_hash = []
        self.file_names = []
        self.fingerprinted_files = []
        self.multiprocessing = multiprocessing
        self.total_fingerprints = 0
        self.auto_save = False

        if len(args) > 0:
            self.load_fingerprinted_files(args[0])

    def save_fingerprinted_files(self, filename):
        data = [self.fingerprinted_files, self.total_fingerprints, self.file_names]
        if filename.split(".")[-1] == "pickle":
            with open(filename, "wb") as f:
                pickle.dump(data, f)
        elif filename.split(".")[-1] == "json":
            with open(filename, "w") as f:
                json.dump(data, f)
        else:
            print("File type must be either pickle or json")

    def load_fingerprinted_files(self, filename):
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
        except FileNotFoundError:
            print(f'"{filename}" not found')

    def fingerprint_directory(
        self, path, plot=False, nprocesses=None, extensions=["*"]
    ):

        # print(f"{pool} : {nprocesses}")
        filenames_to_fingerprint = []
        for filename, _ in decoder.find_files(path, extensions):
            file_name, extension = os.path.splitext(os.path.basename(filename))
            file_name += extension
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

                for processed_file in result:
                    if processed_file[0] != None:
                        if processed_file[0] not in self.file_names:
                            self.fingerprinted_files.append(processed_file)
                            self.file_names.append(processed_file[0])
                            self.total_fingerprints += len(processed_file[1])

        else:

            for filename in filenames_to_fingerprint:
                try:
                    file_name, extension = os.path.splitext(os.path.basename(filename))
                    file_name += extension
                    if file_name in self.file_names:
                        print(f"{file_name} already fingerprinted, continuing...")
                        continue
                    file_name, hashes, file_hash = _fingerprint_worker(
                        filename, limit=self.limit, plot=plot
                    )
                    if file_name != None:
                        self.fingerprinted_files.add([file_name, hashes, file_hash])
                        self.file_names.append(file_name)
                        self.total_fingerprints += len(hashes)
                except:
                    print("Failed fingerprinting")
                    # Print traceback because we can't reraise it here
                    traceback.print_exc(file=sys.stdout)

    def fingerprint_file(self, file_path, set_file_name=None, plot=False):

        file_name, extension = os.path.splitext(os.path.basename(file_path))
        file_name += extension
        if file_name in self.file_names:
            print(f"{file_name} already fingerprinted")
            return

        file_name, hashes, file_hash = self._fingerprint_worker(
            file_path, limit=self.limit, plot=plot
        )
        filename = decoder.path_to_filename(file_path)
        file_name = set_file_name or filename
        if file_name != None:
            self.fingerprinted_files.add([file_name, hashes, file_hash])
            self.file_names.append(file_name)
            self.total_fingerprints += len(hashes)

    def recognize(self, *options, **kwoptions):
        if "recognizer" not in kwoptions.keys():
            r = recognize.FileRecognizer(self)
        elif kwoptions["recognizer"].lower() == "filerecognizer":
            r = recognize.FileRecognizer(self)
            kwoptions.pop("recognizer")
        return r.recognize(*options, **kwoptions)

    def write_processed_file(self, file_name, destination_path):
        decoder.read(file_name, wrdestination=destination_path)

    def plot(self, file_path):
        _fingerprint_worker(file_path, plot=True)


def _fingerprint_worker(file_path, limit=None, plot=False):

    file_name, extension = os.path.splitext(os.path.basename(file_path))
    file_name += extension

    try:
        channel, Fs, file_hash = decoder.read(file_path, limit)
    except FileNotFoundError:
        print(f'"{file_path}" not found')
        return None, None, None
    except:
        print(f'File "{file_name}" could not be decoded')
        return None, None, None

    print(f"Fingerprinting {file_name}")
    hashes = fingerprint.fingerprint(channel, Fs=Fs, plot=plot)
    print(f"Finished fingerprinting {file_name}")

    return file_name, hashes, file_hash


def chunkify(lst, n):
    """
    Splits a list into roughly n equal parts.
    http://stackoverflow.com/questions/2130016/splitting-a-list-of-arbitrary-size-into-only-roughly-n-equal-parts
    """
    return [list(lst)[i::n] for i in range(n)]
