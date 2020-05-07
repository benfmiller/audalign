import os
import sys
import subprocess
import shutil
import zipfile
import calendar
import time
import multiprocessing
from multiprocessing import Pool
import pydub
from collections import defaultdict
import json
import dejavu

TEMP_SUBDIR = 'temp'
TIMEOUT = 10
POLL_INTERVAL = 0.2

def get_hashes(filepath, limit=None):
    #print("Fingerprinting " + filepath + " ...")
    try:
        #Suppress fingerprinting output... too many files...
        devnull = open(os.devnull, 'w')
        sys.stdout = devnull
        (_, song_hashes_list, _) = dejavu._fingerprint_worker(filepath, limit=limit) 
        sys.stdout = sys.__stdout__
    except pydub.exceptions.CouldntDecodeError:
        sys.stdout = sys.__stdout__
        print("Corrupt or nonexistent wav output...")
        return None
    #Converting things to dicts should greatly speedup lookup time when
    #comparing fingerprints...
    song_hashes_dict = defaultdict(list)
    for k, v in song_hashes_list:
        song_hashes_dict[k].append(v) 
    return song_hashes_dict

def to_wav(filepath):
    devnull = open(os.devnull, 'w')
    #print("Converting " + filepath + " to wav...")
    filename, ext = os.path.splitext(filepath)
    #prevent collisions after soundconverter clobbers extension--this can happen
    #under multiprocessing so we rename the file to include the extension before
    #the '.', e.g. foo.xm becomes foo_xm.xm -> foo_xm.wav
    new_filepath = filename + '_' + ext[1:] + ext 
    os.rename(filepath, new_filepath)
    sc_process = subprocess.Popen(["soundconverter", "-b", "-m", "audio/x-wav", "-s", ".wav",\
    new_filepath], stdout=devnull)
    sleep_time = 0
    wavpath = os.path.splitext(new_filepath)[0] + '.wav'
    while sleep_time < TIMEOUT:
        if sc_process.poll() is not None:
            return wavpath
        sleep_time = sleep_time + POLL_INTERVAL
        time.sleep(POLL_INTERVAL)
    print("soundconverter hanging, abandoning conversion for \
    {0}...".format(filepath))
    #Cleanup in case soundconverter took a HUGE dump (sometimes > 50GiB!!!)
    if os.path.exists(wavpath):
        os.remove(wavpath)
    sc_process.kill()
    return None

#Slimmed down version of the same thing from the dejavu library--look ma, no
#mysql db
def count_align_matches(target_hashes, current_hashes):
    diff_counter = {}
    largest_count = 0
    for t_key in target_hashes:
        if t_key in current_hashes:
            for t_offset in target_hashes[t_key]:
                for c_offset in current_hashes[t_key]:
                    diff = t_offset - c_offset
                    if diff not in diff_counter:
                        diff_counter[diff] = 0
                    diff_counter[diff] = diff_counter[diff] + 1
                    if diff_counter[diff] > largest_count:
                        largest_count = diff_counter[diff]
    return largest_count

def process_file_one(arg):
    root = arg[0]
    filename = arg[1]
    target_hashes = arg[2]
    if filename[-4:].lower() != ".zip" and filename[-4:].lower() !=\
    ".wav":
        print("Checking " + root + '/' + filename + "...")
        wavpath = to_wav(root + '/' + filename)
        if wavpath is None:
            return (filename, None)
        current_hashes = get_hashes(wavpath, limit=arg[3])
        if current_hashes is None:
            return (filename, None)
        count = count_align_matches(target_hashes, current_hashes)
        #prevent collisions where two mods have different extensions
        #but same name--leading to colliding wavs 
        if os.path.exists(wavpath):
            os.remove(wavpath)
        return (filename, count)
    #Leftover wav, will be ignored
    else:
        return ('', 0)

class ModSearch:
    def __init__(self, target_path, search_path=None, temp_path=None, duration_limit=None):
        #Keep track of how hideously long this takes
        self.START_TIME = calendar.timegm(time.gmtime())
        self.SEARCHED = 0
        self.search_path = '/home/ketsol/mods/'
        self.temp_path = './temp' + '/' + TEMP_SUBDIR
        print("Fingerprinting Target...")
        self.target_hashes = get_hashes(target_path)
        self.largest_counts = {}
        self.checked_fileset = set()
        self.mp = True
        self.p_count = multiprocessing.cpu_count()
        self.duration_limit = 30
        if os.path.exists(self.search_path + 'files_log'):
            with open(self.search_path + 'files_log', 'r') as files_log:
                self.checked_fileset = set(files_log.read().splitlines())
        if os.path.exists(self.search_path + 'match_log'):
            with open(self.search_path + 'match_log', 'r') as match_log:
                self.largest_counts = json.load(match_log) 
            print('Loaded match log ' + str(self.largest_counts))   
   
        if search_path is not None: 
            self.search_path = search_path
            assert(temp_path is not None)
            if temp_path[-1] == '/':
                self.temp_path = temp_path + TEMP_SUBDIR
            else:
                self.temp_path = temp_path + '/' + TEMP_SUBDIR
        if duration_limit is not None:
            self.duration_limit = int(duration_limit)

    #Unzip all of the nested zips in a directory, without regard for flattening
    #their directory structures... we just want the mods!
    def unzip_all(self):
        zipfiles = set()
        for root, dirs, files in os.walk(self.temp_path):
            for filename in files:
                #print(filename)
                if filename[-4:].lower() == '.zip':
                   zipfiles.add(root + '/' + filename) 
        while zipfiles:
            for filepath in zipfiles:
                #print(filepath)
                try: 
                    with zipfile.ZipFile(filepath, 'r') as cur_zip:
                        cur_zip.extractall(self.temp_path)
                except zipfile.BadZipfile:
                    print("Bad zip encountered at {0}...".format(filepath))
                    os.remove(filepath)
                    continue
                os.remove(filepath)
            zipfiles = set()
            for root, dirs, files in os.walk(self.temp_path):
                for filename in files:
                    if filename[-4:].lower() == '.zip':
                        zipfiles.add(root + '/' + filename)  

    def process_dir(self):
        self.process_mods()
    
    #Copy all of the zips from the search directory to the temporary directory
    #and process them
    def process_all_zips(self):
        if self.mp:
            self.p = Pool(self.p_count)
        print(self.search_path)
        for root, dirs, files in os.walk(self.search_path):
            for filename in files:
                if filename[-4:].lower() == '.zip':
                    if not os.path.exists(self.temp_path):
                        os.makedirs(self.temp_path)
                    shutil.copy(root+'/'+filename, self.temp_path + '/' + filename)
                    self.unzip_all()
                    self.process_dir()
                    shutil.rmtree(self.temp_path)
        print(sorted(self.largest_counts.items(), key=lambda match: match[1]))
        if self.mp:
            self.p.close()
            self.p.join()

    def process_file(self, root, filename):
        if filename in self.checked_fileset:
            return None
        if filename[-4:].lower() != ".zip" and filename[-4:].lower() !=\
        ".wav":
            #print("Checking " + root + '/' + filename + "...")
            wavpath = to_wav(root + '/' + filename)
            if wavpath is None:
                return None
            current_hashes = get_hashes(wavpath, limit=self.duration_limit)
            if current_hashes is None:
                return None
            count = count_align_matches(self.target_hashes, current_hashes)
            #prevent collisions where two mods have different extensions
            #but same name--leading to colliding wavs 
            if os.path.exists(wavpath):
                os.remove(wavpath)
            return count
    
    #check if match count is above threshold, log checked files and matches
    def process_result(self, count, filename):
        if count is not None and count > 10:
            self.largest_counts[filename] = count
        if len(self.largest_counts.items()) > 0:
            print("Current Top Matches Are " +\
            str(sorted(self.largest_counts.items(), key=lambda match:\
            match[1])) + " ...")
        with open(self.search_path + 'files_log', 'a') as files_log:
            files_log.write(filename + '\n')
        with open(self.search_path + 'match_log', 'w') as match_log:
            json.dump(self.largest_counts, match_log)
    
    #After all of the mods in the temp directory have been extracted, process
    #them
    def process_mods(self):
        counts = {}
        count = 0
        for root, dirs, files in os.walk(self.temp_path):
            if not self.mp:
                for filename in files:
                    self.SEARCHED = self.SEARCHED + 1
                    count = self.process_file(root, filename)
                    cur_time = calendar.timegm(time.gmtime())
                    print("Searched {0:d} mods in {1:d} seconds...".format(self.SEARCHED,\
                    cur_time - self.START_TIME))
                    self.process_result(count, filename)
            else:
                #Pack object state into a list... pretty nasty, feels like
                #OpenCL stuff
                map_input = [[root, filename,
                self.target_hashes, self.duration_limit]\
                for filename in files if filename not in\
                self.checked_fileset]
                results = self.p.map(process_file_one, map_input)
                self.SEARCHED = self.SEARCHED + len(files)
                cur_time = calendar.timegm(time.gmtime())
             
                print("Searched {0:d} mods in {1:d} seconds...".format(self.SEARCHED,\
                cur_time - self.START_TIME))
                print(results)
                for result in results:
                    count = result[1]
                    filename = result[0]
                    self.process_result(count, filename)

def main():
    if len(sys.argv) > 1:
        assert(len(sys.argv) > 3)
        if len(sys.argv) > 4:
            m_modsearcher = ModSearch(sys.argv[1], sys.argv[2], sys.argv[3],\
            sys.argv[4])
        else:
            m_modsearcher = ModSearch(sys.argv[1], sys.argv[2], sys.argv[3])
    else: 
        m_modsearcher = ModSearch("./test/2.wav")
    m_modsearcher.process_all_zips()    
    

if __name__ == "__main__":
    main()
