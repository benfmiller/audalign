import os
import fnmatch
import numpy as np
from pydub import AudioSegment


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


def read(filename, wrdestination=None):
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
        with open(wrdestination, "wb") as file_place:
            audiofile.export(file_place, format=os.path.splitext(file_place)[1])

    return data, audiofile.frame_rate


def shift_write_files(files_shifts, destination_path, names_and_paths, write_extension):

    cant_write_ext = [".mov", ".mp4"]

    if write_extension:
        if write_extension[0] != ".":
            write_extension = "." + write_extension

    audsegs = []
    for name in files_shifts.keys():
        file_path = names_and_paths[name]

        silence = AudioSegment.silent(files_shifts[name] * 1000, frame_rate=44100)

        audiofile = AudioSegment.from_file(file_path)

        audiofile = audiofile.set_frame_rate(44100)
        audiofile = audiofile.set_sample_width(2)
        audiofile = audiofile.set_channels(1)
        audiofile = audiofile.normalize()

        file_name = os.path.basename(file_path)
        destination_name = os.path.join(destination_path, file_name)

        audiofile = silence + audiofile

        if os.path.splitext(destination_name)[1] in cant_write_ext:
            destination_name = os.path.splitext(destination_name)[0] + ".wav"

        if write_extension:
            destination_name = os.path.join(
                os.path.splitext(destination_name)[0], write_extension
            )

            with open(destination_name, "wb") as file_place:
                audiofile.export(
                    file_place, format=os.path.splitext(destination_name)[1][1:]
                )

        else:
            with open(destination_name, "wb") as file_place:
                audiofile.export(
                    file_place, format=os.path.splitext(destination_name)[1][1:]
                )

        audsegs += [audiofile]

    total_files = audsegs[0]

    for i in audsegs[1:]:
        total_files = total_files.overlay(i)

    if write_extension:

        total_name = os.path.join(destination_path, "total", write_extension)

        with open(total_name, "wb") as file_place:
            total_files.export(file_place, format=os.path.splitext(total_name)[1][1:])

    else:

        total_name = os.path.join(destination_path, "total.wav")

        with open(total_name, "wb") as file_place:
            total_files.export(file_place, format=os.path.splitext(total_name)[1][1:])


def shift_write_file(file_path, destination_path, offset_seconds):

    silence = AudioSegment.silent(offset_seconds * 1000, frame_rate=44100)

    audiofile = AudioSegment.from_file(file_path)

    audiofile = audiofile.set_frame_rate(44100)
    audiofile = audiofile.set_sample_width(2)
    audiofile = audiofile.set_channels(1)
    audiofile = audiofile.normalize()

    audiofile = silence + audiofile

    with open(destination_path, "wb") as file_place:
        audiofile.export(file_place, format=os.path.splitext(destination_path)[1][1:])


def convert_audio_file():
    pass
