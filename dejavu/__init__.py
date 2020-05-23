#from dejavu.database import get_database, Database
import dejavu.decoder as decoder
import dejavu.fingerprint as fingerprint
import multiprocessing
import os
import traceback
import sys
import pickle
import json


class Dejavu(object):

    SONG_ID = "song_id"
    SONG_NAME = 'song_name'
    CONFIDENCE = 'confidence'
    MATCH_TIME = 'match_time'                             
    OFFSET = 'offset'
    OFFSET_SECS = 'offset_seconds'

    def __init__(self, *args): #, config):
        #super(Dejavu, self).__init__()

        self.limit = None
        self.file_unique_hash = []
        self.fingerprinted_files = []
        if len(args) > 0:
            self.get_fingerprinted_songs(args[0])
        

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
        
        
    def save_fingerprinted_songs(self, filename):
        if filename.split('.')[-1] == 'pickle':
            with open(filename, 'wb') as f:
                pickle.dump(self.fingerprinted_files, f)
        elif filename.split('.')[-1] == 'json':
            with open(filename, 'w') as f:
                json.dump(self.fingerprinted_files, f)
        else:
            print('File type must be either pickle or json')

    def get_fingerprinted_songs(self, filename):
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

    def fingerprint_directory(self, path, extensions, nprocesses=None):
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

        # Prepare _fingerprint_worker input
        worker_input = zip(filenames_to_fingerprint,
                           [self.limit] * len(filenames_to_fingerprint))

        list_song_and_hashes = []

        for i in worker_input:
            try:
                song_name, hashes, file_hash = _fingerprint_worker(i)
                list_song_and_hashes += [[song_name, hashes, file_hash]]
            except:
                print("Failed fingerprinting")
                # Print traceback because we can't reraise it here
                traceback.print_exc(file=sys.stdout)
                
        
        self.fingerprinted_files += list_song_and_hashes

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

    def fingerprint_file(self, filepath, song_name=None):
        songname = decoder.path_to_songname(filepath)
        song_hash = decoder.unique_hash(filepath)
        song_name = song_name or songname
        # don't refingerprint already fingerprinted files
        if song_hash in self.file_unique_hash:
            print ("%s already fingerprinted, continuing..." % song_name)
        else:
            song_name, hashes, file_hash = _fingerprint_worker(
                filepath,
                self.limit,
                song_name=song_name
            )
            self.fingerprinted_files += [[song_name, hashes, file_hash]]

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
        largest = 0
        largest_count = 0
        song_name = -1
        for pair in matches:
            sid, diff = pair
            if diff not in diff_counter:
                diff_counter[diff] = {}
            if sid not in diff_counter[diff]:
                diff_counter[diff][sid] = 0
            diff_counter[diff][sid] += 1

            if diff_counter[diff][sid] > largest_count:
                largest = diff
                largest_count = diff_counter[diff][sid]
                song_name = sid

        # extract idenfication
        song_id = self.get_file_id(song_name)

        # return match info
        nseconds = round(float(largest) / fingerprint.DEFAULT_FS *
                         fingerprint.DEFAULT_WINDOW_SIZE *
                         fingerprint.DEFAULT_OVERLAP_RATIO, 5)
        song = {
            Dejavu.SONG_ID : song_id,
            Dejavu.SONG_NAME : song_name,
            Dejavu.CONFIDENCE : largest_count,
            Dejavu.OFFSET : int(largest),
            Dejavu.OFFSET_SECS : nseconds,
            #Database.FIELD_FILE_SHA1 : song.get(Database.FIELD_FILE_SHA1, None).encode("utf8"),
            }
        return song

    def recognize(self, recognizer, *options, **kwoptions):
        r = recognizer(self)
        return r.recognize(*options, **kwoptions)

    def get_file_id(self, name):
        for i in self.fingerprinted_files:
            if i[0] == name:
                return i[2]


def _fingerprint_worker(filename, limit=None, song_name=None):
    # Pool.imap sends arguments as tuples so we have to unpack
    # them ourself.
    try:
        filename, limit = filename
    except ValueError:
        pass

    songname, extension = os.path.splitext(os.path.basename(filename))
    song_name = song_name or songname
    channels, Fs, file_hash = decoder.read(filename, limit)
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

    return song_name, result, file_hash


def chunkify(lst, n):
    """
    Splits a list into roughly n equal parts.
    http://stackoverflow.com/questions/2130016/splitting-a-list-of-arbitrary-size-into-only-roughly-n-equal-parts
    """
    return [lst[i::n] for i in range(n)]
