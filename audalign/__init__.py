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
from pydub import AudioSegment


class Audalign(object):

    FILE_ID = "file_id"
    FILE_NAME = "file_name"
    CONFIDENCE = "confidence"
    MATCH_TIME = "match_time"
    OFFSET_SAMPLES = "offset_samples"
    OFFSET_SECS = "offset_seconds"

    def __init__(self, *args, multiprocessing=True):  # , config):
        # super(audalign, self).__init__()

        self.limit = None
        self.file_unique_hash = []
        self.fingerprinted_files = []
        self.multiprocessing = multiprocessing

        if len(args) > 0:
            self.get_fingerprinted_files(args[0])

        # --------------------------------------------------

        # self.config = config

        # initialize db
        # db_cls = get_database(config.get("database_type", None))

        # self.db = db_cls(**config.get("database", {}))
        # self.db.setup()

        # ---------------------------------------------

        # if we should limit seconds fingerprinted,
        # None|-1 means use entire track
        # self.limit = self.config.get("fingerprint_limit", None)
        # if self.limit == -1:  # for JSON compatibility
        #    self.limit = None

    def save_fingerprinted_files(self, filename):
        if filename.split(".")[-1] == "pickle":
            with open(filename, "wb") as f:
                pickle.dump(self.fingerprinted_files, f)
        elif filename.split(".")[-1] == "json":
            with open(filename, "w") as f:
                json.dump(self.fingerprinted_files, f)
        else:
            print("File type must be either pickle or json")

    def get_fingerprinted_files(self, filename):
        try:
            if filename.split(".")[-1] == "pickle":
                with open(filename, "rb") as f:
                    self.fingerprinted_files += pickle.load(f)
            elif filename.split(".")[-1] == "json":
                with open(filename, "r") as f:
                    self.fingerprinted_files += json.load(f)
            else:
                print("File type must be either pickle or json")
        except FileNotFoundError:
            print(f'"{filename}" not found')


    def fingerprint_directory(self, path, filt_name=False, plot=False, nprocesses=None):

        # print(f"{pool} : {nprocesses}")
        filenames_to_fingerprint = []
        for filename, _ in decoder.find_files(path, ["*"]):

            # don't refingerprint already fingerprinted files
            if decoder.unique_hash(filename) in self.file_unique_hash:
                print("%s already fingerprinted, continuing..." % filename)
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
                    if processed_file[2] != None:
                        self.fingerprinted_files.append(processed_file)

        else:

            for filename in filenames_to_fingerprint:
                try:
                    file_name, hashes, file_hash = _fingerprint_worker(
                        filename, limit=self.limit, file_name=None, plot=plot
                    )
                    if file_name != None:
                        self.fingerprinted_files.append([file_name, hashes, file_hash])
                except:
                    print("Failed fingerprinting")
                    # Print traceback because we can't reraise it here
                    traceback.print_exc(file=sys.stdout)

    def fingerprint_file(self, file_path, file_name=None, plot=False):
        filename = decoder.path_to_filename(file_path)
        try:
            file_hash = decoder.unique_hash(file_path)
        except FileNotFoundError:
            print(f'"{file_path}" not found')
            return
        file_name = file_name or filename
        # don't refingerprint already fingerprinted files
        if file_hash in self.file_unique_hash:
            print("%s already fingerprinted, continuing..." % file_name)
        else:
            file_name, hashes, file_hash = _fingerprint_worker(
                file_path, self.limit, file_name, plot
            )
            if file_hash != None:
                self.fingerprinted_files += [[file_name, hashes, file_hash]]

    def find_matches(self, samples, Fs=fingerprint.DEFAULT_FS):
        target_mapper = fingerprint.fingerprint(samples, Fs=Fs)
        matches = []

        for audio_file in self.fingerprinted_files:
            already_hashes = audio_file[1]
            for channel in already_hashes:
                for t_hash in target_mapper.keys():
                    if t_hash in already_hashes.keys():
                        for t_offset in target_mapper[t_hash]:
                            for a_offset in already_hashes[t_hash]:
                                diff = a_offset - t_offset
                                matches.append([audio_file[0], diff])
        return matches

    def align_matches(self, matches):
        """
            Finds hash matches that align in time with other matches and finds
            consensus about which hashes are "true" signal from the audio.

            Returns a dictionary with match information.
        """
        # align by diffs
        diff_counter = {}
        largest_match_offset = 0
        largest_match_count = 0
        file_name = -1
        for pair in matches:
            sid, diff = pair
            if diff not in diff_counter:
                diff_counter[diff] = {}
            if sid not in diff_counter[diff]:
                diff_counter[diff][sid] = 0
            diff_counter[diff][sid] += 1

            if diff_counter[diff][sid] > largest_match_count:
                largest_match_offset = diff
                largest_match_count = diff_counter[diff][sid]
                file_name = sid

        # extract idenfication
        file_id = self.get_file_id(file_name)

        # return match info
        nseconds = round(
            float(largest_match_offset)
            / fingerprint.DEFAULT_FS
            * fingerprint.DEFAULT_WINDOW_SIZE
            * fingerprint.DEFAULT_OVERLAP_RATIO,
            5,
        )
        audio_file = {
            Audalign.FILE_ID: file_id,
            Audalign.FILE_NAME: file_name,
            Audalign.CONFIDENCE: largest_match_count,
            Audalign.OFFSET_SAMPLES: int(largest_match_offset),
            Audalign.OFFSET_SECS: nseconds,
            # Database.FIELD_FILE_SHA1 : song.get(Database.FIELD_FILE_SHA1, None).encode("utf8"),
        }
        return audio_file

    def recognize(self, *options, **kwoptions):
        if "recognizer" not in kwoptions.keys():
            r = recognize.FileRecognizer(self)
        elif kwoptions["recognizer"].lower() == "microphonerecognizer":
            r = recognize.MicrophoneRecognizer(self)
            kwoptions.pop("recognizer")
        elif kwoptions["recognizer"].lower() == "filerecognizer":
            r = recognize.FileRecognizer(self)
            kwoptions.pop("recognizer")
        return r.recognize(*options, **kwoptions)

    def get_file_id(self, name):
        for i in self.fingerprinted_files:
            if i[0] == name:
                return i[2]

    def write_processed_file(self, file_name, destination_path):
        channel, Fs, file_hash = decoder.read(file_name, wrdestination=destination_path)

def _fingerprint_worker(file_path, limit=None, file_name=None, plot=False):
    # Pool.imap sends arguments as tuples so we have to unpack
    # them ourself.
    try:
        file_path, limit = file_path
    except ValueError:
        pass

    file_name, extension = os.path.splitext(os.path.basename(file_path))
    file_name += extension

    try:
        channel, Fs, file_hash = decoder.read(file_path, limit)
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
