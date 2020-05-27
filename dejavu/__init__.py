#from dejavu.database import get_database, Database
import dejavu.decoder as decoder
import dejavu.fingerprint as fingerprint
import dejavu.recognize as recognize
import multiprocessing
import os
import traceback
import sys
import pickle
import json


class Dejavu(object):

    FILE_ID = "file_id"
    FILE_NAME = 'file_name'
    CONFIDENCE = 'confidence'
    MATCH_TIME = 'match_time'                             
    OFFSET_SAMPLES = 'offset_samples'
    OFFSET_SECS = 'offset_seconds'

    def __init__(self, *args): #, config):
        #super(Dejavu, self).__init__()

        self.limit = None
        self.file_unique_hash = []
        self.fingerprinted_files = []
        if len(args) > 0:
            self.get_fingerprinted_files(args[0])
        

        #--------------------------------------------------

        #self.config = config

        # initialize db
        #db_cls = get_database(config.get("database_type", None))

        #self.db = db_cls(**config.get("database", {}))
        #self.db.setup()

        #---------------------------------------------

        # if we should limit seconds fingerprinted,
        # None|-1 means use entire track
        #self.limit = self.config.get("fingerprint_limit", None)
        #if self.limit == -1:  # for JSON compatibility
        #    self.limit = None
        
        
    def save_fingerprinted_files(self, filename):
        if filename.split('.')[-1] == 'pickle':
            with open(filename, 'wb') as f:
                pickle.dump(self.fingerprinted_files, f)
        elif filename.split('.')[-1] == 'json':
            with open(filename, 'w') as f:
                json.dump(self.fingerprinted_files, f)
        else:
            print('File type must be either pickle or json')

    def get_fingerprinted_files(self, filename):
        if filename.split('.')[-1] == 'pickle':
            with open(filename, 'rb') as f:
                self.fingerprinted_files = pickle.load(f)
        elif filename.split('.')[-1] == 'json':
            with open(filename, 'r') as f:
                self.fingerprinted_files = json.load(f)
        else:
            print('File type must be either pickle or json')
            
    """
        # get songs previously indexed
        self.songs = self.db.get_songs()
        self.file_unique_hash = set()  # to know which ones we've computed before
        for song in self.songs:
            song_hash = song[Database.FIELD_FILE_SHA1]
            self.file_unique_hash.add(song_hash)"""

    def fingerprint_directory(self, path, extensions=[".mp3", ".wav", ".mp4", ".MOV"], nprocesses=None):
        # Try to use the maximum amount of processes if not given.
        """try:
            #nprocesses = nprocesses or multiprocessing.cpu_count()
            nprocesses = 1
        except NotImplementedError:
            nprocesses = 1
        else:
            nprocesses = 1 if nprocesses <= 0 else nprocesses"""

        #pool = multiprocessing.Pool(nprocesses)

        filenames_to_fingerprint = []
        for filename, _ in decoder.find_files(path, extensions):

            # don't refingerprint already fingerprinted files
            if decoder.unique_hash(filename) in self.file_unique_hash:
                print ("%s already fingerprinted, continuing..." % filename)
                continue

            filenames_to_fingerprint.append(filename) 

        #TODO: Redo preparation
        # Prepare _fingerprint_worker input
        worker_input = zip(filenames_to_fingerprint,
                           [self.limit] * len(filenames_to_fingerprint))

        list_file_and_hashes = []

        for i in worker_input:
            try:
                file_name, hashes, file_hash = _fingerprint_worker(i)
                if file_name != None:
                    list_file_and_hashes += [[file_name, hashes, file_hash]]
            except:
                print("Failed fingerprinting")
                # Print traceback because we can't reraise it here
                traceback.print_exc(file=sys.stdout)
                
        
        self.fingerprinted_files += list_file_and_hashes

        """
        # Send off our tasks
        iterator = pool.izip(_fingerprint_worker,
                                       worker_input)

        # Loop till we have all of them
        while True:
            try:
                song_name, hashes, file_hash = iterator.next()
            except multiprocessing.TimeoutError:
                continue
            except StopIteration:
                break
            except:
                print("Failed fingerprinting")
                # Print traceback because we can't reraise it here
                traceback.print_exc(file=sys.stdout)
            else:
                sid = self.db.insert_song(song_name, file_hash)

                self.db.insert_hashes(sid, hashes)
                self.db.set_song_fingerprinted(sid)
                self.get_fingerprinted_songs()

        pool.close()
        pool.join()
        """

    def fingerprint_file(self, filepath, normalize=True, file_name=None):
        filename = decoder.path_to_filename(filepath)
        file_hash = decoder.unique_hash(filepath)
        file_name = file_name or filename
        # don't refingerprint already fingerprinted files
        if file_hash in self.file_unique_hash:
            print ("%s already fingerprinted, continuing..." % file_name)
        else:
            file_name, hashes, file_hash = _fingerprint_worker(
                filepath,
                self.limit,
                normalize,
                file_name=file_name
            )
            if file_hash != None:
                self.fingerprinted_files += [[file_name, hashes, file_hash]]

            """
            sid = self.db.insert_song(song_name, file_hash)

            self.db.insert_hashes(sid, hashes)
            self.db.set_song_fingerprinted(sid)
            self.get_fingerprinted_songs()"""

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
        nseconds = round(float(largest_match_offset) / fingerprint.DEFAULT_FS *
                         fingerprint.DEFAULT_WINDOW_SIZE *
                         fingerprint.DEFAULT_OVERLAP_RATIO, 5)
        audio_file = {
            Dejavu.FILE_ID : file_id,
            Dejavu.FILE_NAME : file_name,
            Dejavu.CONFIDENCE : largest_match_count,
            Dejavu.OFFSET_SAMPLES : int(largest_match_offset),
            Dejavu.OFFSET_SECS : nseconds,
            #Database.FIELD_FILE_SHA1 : song.get(Database.FIELD_FILE_SHA1, None).encode("utf8"),
            }
        return audio_file

    def recognize(self, *options, **kwoptions):
        if 'recognizer' not in kwoptions.keys():
            r = recognize.FileRecognizer(self)
        elif kwoptions['recognizer'].lower() == 'microphonerecognizer':
            r = recognize.MicrophoneRecognizer(self)
            kwoptions.pop('recognizer')
        elif kwoptions['recognizer'].lower() == 'filerecognizer':
            r = recognize.FileRecognizer(self)
            kwoptions.pop('recognizer')
        return r.recognize(*options, **kwoptions)

    def get_file_id(self, name):
        for i in self.fingerprinted_files:
            if i[0] == name:
                return i[2]


def _fingerprint_worker(filename, normalize=True, limit=None, file_name=None):
    # Pool.imap sends arguments as tuples so we have to unpack
    # them ourself.
    try:
        filename, limit = filename
    except ValueError:
        pass

    filename, extension = os.path.splitext(os.path.basename(filename))
    file_name = file_name or filename
    try:
        channels, Fs, file_hash = decoder.read(filename, normalize, limit)
    except:
        print(f"File \"{filename + extension}\" could not be decoded")
        return None, None, None
    result = {}
    channel_amount = len(channels)

    for channeln, channel in enumerate(channels):
        # TODO: Remove prints or change them into optional logging.
        print("Fingerprinting channel %d/%d for %s" % (channeln + 1,
                                                       channel_amount,
                                                       filename))
        hashes = fingerprint.fingerprint(channel, Fs=Fs)
        print("Finished channel %d/%d for %s" % (channeln + 1, channel_amount,
                                                 filename))
        for hash_ in hashes.keys():
            if hash_ not in result.keys():
                result[hash_] = hashes[hash_]
            else:
                result[hash_] += hashes[hash_]

    return file_name, result, file_hash


def chunkify(lst, n):
    """
    Splits a list into roughly n equal parts.
    http://stackoverflow.com/questions/2130016/splitting-a-list-of-arbitrary-size-into-only-roughly-n-equal-parts
    """
    return [lst[i::n] for i in range(n)]
