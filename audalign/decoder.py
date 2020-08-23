import os
import fnmatch
import numpy as np
from pydub import AudioSegment
from pydub.utils import audioop
from hashlib import sha1


def find_files(path, extensions=["*"]):
    """
    Yields all files with given extension in path and all subdirectories

    Parameters
    ----------
    path : str
        path to folder
    extensions : list[str]
        list of all extensions to include

    Yields
    ------
    p : str
        file path
    extension : str
        extension of file
    """

    for dirpath, dirnames, files in os.walk(path):
        for extension in extensions:
            for f in fnmatch.filter(files, "*.%s" % extension):
                p = os.path.join(dirpath, f)
                yield (p, extension)


def read(filename, wrdestination=None, adjust_alignment=None):
    """
    Reads any file supported by pydub (ffmpeg) and returns a numpy array and the bit depth

    Parameters
    ----------
    filename : str
        path to audio file
    wrdestination : str
        writes the audio file after processing

    Returns
    -------
    channel : array[int]
        array of audio data
    frame_rate : int
        returns the bit depth
    """

    audiofile = AudioSegment.from_file(filename)

    audiofile = audiofile.set_frame_rate(44100)
    audiofile = audiofile.set_sample_width(2)
    audiofile = audiofile.set_channels(1)
    audiofile = audiofile.normalize()

    data = np.frombuffer(audiofile._data, np.int16)

    if wrdestination:
        audiofile.export(wrdestination)

    if adjust_alignment:
        shift_file(audiofile, adjust_alignment)

    return data, audiofile.frame_rate

def shift_file(audiofile, adjustment):
    pass # not implemented yet
