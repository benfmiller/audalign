import os
import fnmatch
import numpy as np
from pydub import AudioSegment
from pydub.utils import audioop
import wavio
from hashlib import sha1


def unique_hash(filepath, blocksize=2 ** 20):
    """ Small function to generate a hash to uniquely generate
    a file. Inspired by MD5 version here:
    http://stackoverflow.com/a/1131255/712997

    Works with large files. 
    """
    s = sha1()
    with open(filepath, "rb") as f:
        while True:
            buf = f.read(blocksize)
            if not buf:
                break
            s.update(buf)
    return s.hexdigest().upper()


def find_files(path, extensions):
    # Allow both with ".mp3" and without "mp3" to be used for extensions
    # extensions = [e.replace(".", "") for e in extensions]

    for dirpath, dirnames, files in os.walk(path):
        for extension in extensions:
            for f in fnmatch.filter(files, "*.%s" % extension):
                p = os.path.join(dirpath, f)
                yield (p, extension)


def read(filename, limit=None):
    """
    Reads any file supported by pydub (ffmpeg) and returns the data contained
    within. If file reading fails due to input being a 24-bit wav file,
    wavio is used as a backup.

    Can be optionally limited to a certain amount of seconds from the start
    of the file by specifying the `limit` parameter. This is the amount of
    seconds from the start of the file.

    returns: (channels, samplerate)
    """
    
    audiofile = AudioSegment.from_file(filename)
    
    audiofile = audiofile.set_frame_rate(44100)
    audiofile = audiofile.set_sample_width(2)
    audiofile = audiofile.set_channels(1)
    audiofile = audiofile.normalize()

    if limit:
        audiofile = audiofile[: limit * 1000]

    data = np.fromstring(audiofile._data, np.int16)

    channels = data[1 :: audiofile.channels]
    """for chn in range(audiofile.channels):
        channels.append(data[chn :: audiofile.channels])"""

    fs = audiofile.frame_rate

    return channels, audiofile.frame_rate, unique_hash(filename)


def path_to_filename(path):
    """
    Extracts file name from a filepath. Used to identify which files
    have already been fingerprinted on disk.
    """
    return os.path.splitext(os.path.basename(path))[0]
