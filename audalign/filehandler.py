import multiprocessing
import os
import fnmatch
import numpy as np
from numpy.core.defchararray import array
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import math
from audalign.fingerprint import DEFAULT_FS
import noisereduce
from functools import partial

cant_write_ext = [".mov", ".mp4"]


def find_files(path, extensions=["*"]):
    """
    Yields all files with given extension in path and all subdirectories

    Args
        path (str): path to folder
        extensions (list[str]): list of all extensions to include

    Yields
        p (str): file path
        extension (str): extension of file
    """

    for dirpath, dirnames, files in os.walk(path):
        for extension in extensions:
            for f in fnmatch.filter(files, "*.%s" % extension):
                p = os.path.join(dirpath, f)
                yield (p, extension)


def create_audiosegment(filepath: str, start_end: tuple = None, sample_rate=DEFAULT_FS):
    if sample_rate is None:
        sample_rate = DEFAULT_FS
    if os.path.splitext(filepath)[1] in [".txt", ".json"]:
        raise CouldntDecodeError
    audiofile = AudioSegment.from_file(filepath)
    audiofile = audiofile.set_frame_rate(sample_rate)
    audiofile = audiofile.set_sample_width(2)
    audiofile = audiofile.set_channels(1)
    audiofile = audiofile.normalize()
    if start_end is not None:

        # Does the preprocessing and bounds checking
        start_end = list(start_end)
        start_end = [start_end[0] * 1000, start_end[1] * 1000]
        if start_end[1] > 0 and start_end[1] < start_end[0]:
            raise ValueError  # if end is greater than 0, end must be greater than start
        if start_end[0] < 0:
            raise ValueError  # Start must be >= 0
        if start_end[0] > len(audiofile):
            start_end[0] = len(audiofile)
        if start_end[1] > len(audiofile):
            start_end[1] = len(audiofile)
        if start_end[1] * -1 > len(audiofile):
            start_end[1] = len(audiofile) * -1

        # Does the silencing for start
        start_silence = AudioSegment.silent(
            duration=(start_end[0]), frame_rate=sample_rate
        )
        audiofile = start_silence + audiofile[start_end[0] :]

        # Does the silencing for end
        if start_end[1] > 0:
            end_silence = AudioSegment.silent(
                duration=len(audiofile) - (start_end[1]), frame_rate=sample_rate
            )
            audiofile = audiofile[: start_end[1]] + end_silence
        elif start_end[1] < 0:
            end_silence = AudioSegment.silent(
                duration=(start_end[1]) * -1, frame_rate=sample_rate
            )
            start_end[1] += len(audiofile)
            audiofile = audiofile[: start_end[1]] + end_silence

    return audiofile


def get_audio_files_directory(directory_path: str) -> list:
    """returns a list of the file paths in directory that are audio

    Args:
        directory_path (str): string of filepath

    Returns:
        list: of all paths in file that are audio
    """
    aud_list = []
    for file_path, _ in find_files(directory_path):
        try:
            if os.path.splitext(file_path)[1] in [".txt", ".json"]:
                continue
            AudioSegment.from_file(file_path)
            aud_list += [os.path.basename(file_path)]
        except CouldntDecodeError:
            pass  # Do nothing
    return aud_list


def read(
    filename: str,
    wrdestination=None,
    start_end: tuple = None,
    sample_rate=DEFAULT_FS,
):
    """
    Reads any file supported by pydub (ffmpeg) and returns a numpy array and the bit depth

    Args
        filename (str): path to audio file
        wrdestination (str): writes the audio file after processing

    Returns
    -------
        channel (array[int]): array of audio data
        frame_rate (int): returns the bit depth
    """

    audiofile = create_audiosegment(
        filename, start_end=start_end, sample_rate=sample_rate
    )
    data = np.frombuffer(audiofile._data, np.int16)
    if wrdestination:
        with open(wrdestination, "wb") as file_place:
            audiofile.export(file_place, format=os.path.splitext(wrdestination)[1][1:])
    return data, audiofile.frame_rate


def _floatify_data(audio_segment: AudioSegment):
    data = np.frombuffer(audio_segment._data, np.int16)
    new_data = np.zeros(len(data))
    for i in range(len(data)):
        if data[i] < 0:
            new_data[i] = float(data[i]) / 32768
        elif data[i] == 0:
            new_data[i] = 0.0
        if data[i] > 0:
            new_data[i] = float(data[i]) / 32767
    return new_data


def _int16ify_data(data: array):
    for i in range(len(data)):
        if data[i] < 0:
            data[i] = int(data[i] * 32768)
        elif data[i] == 0:
            data[i] = int(0)
        else:
            data[i] = int(data[i] * 32767)
    return data


def noise_remove(
    filepath,
    noise_start,
    noise_end,
    destination,
    alt_noise_filepath=None,
    prop_decrease=1,
    use_tensorflow=False,
    verbose=False,
    **kwargs,
):

    audiofile = create_audiosegment(filepath)
    new_data = _floatify_data(audiofile)

    if not alt_noise_filepath:
        noisy_part = new_data[(noise_start * DEFAULT_FS) : (noise_end * DEFAULT_FS)]
    else:
        noise_audiofile = create_audiosegment(alt_noise_filepath)
        noise_new_data = _floatify_data(noise_audiofile)
        noisy_part = noise_new_data[
            (noise_start * DEFAULT_FS) : (noise_end * DEFAULT_FS)
        ]

    reduced_noise_data = noisereduce.reduce_noise(
        new_data,
        noisy_part,
        prop_decrease=prop_decrease,
        use_tensorflow=use_tensorflow,
        verbose=verbose,
        **kwargs,
    )

    reduced_noise_data = _int16ify_data(reduced_noise_data)
    audiofile._data = reduced_noise_data.astype(np.int16)
    with open(destination, "wb") as file_place:
        audiofile.export(file_place, format=os.path.splitext(destination)[1][1:])


def noise_remove_directory(
    directory,
    noise_filepath,
    noise_start,
    noise_end,
    destination_directory,
    prop_decrease=1,
    use_tensorflow=False,
    verbose=False,
    use_multiprocessing=False,
    num_processes=None,
    **kwargs,
):
    noise_data = _floatify_data(create_audiosegment(noise_filepath))[
        (noise_start * DEFAULT_FS) : (noise_end * DEFAULT_FS)
    ]
    file_names = []
    for file_path, _ in find_files(directory):
        file_names += [file_path]

    _reduce_noise = partial(
        _remove_noise,
        noise_section=noise_data,
        destination_directory=destination_directory,
        prop_decrease=prop_decrease,
        use_tensorflow=use_tensorflow,
        verbose=verbose,
        **kwargs,
    )

    if use_multiprocessing == True:

        try:
            nprocesses = num_processes or multiprocessing.cpu_count()
        except NotImplementedError:
            nprocesses = 1
        else:
            nprocesses = 1 if nprocesses <= 0 else nprocesses

        with multiprocessing.Pool(nprocesses) as pool:

            pool.map(_reduce_noise, file_names)

            pool.close()
            pool.join()
    else:
        for i in file_names:
            _reduce_noise(i)


def _remove_noise(
    file_path,
    noise_section=[],
    destination_directory="",
    prop_decrease=1,
    use_tensorflow=False,
    verbose=False,
    **kwargs,
):

    try:
        print(f"Reducing noise: {file_path}")
        audiofile = create_audiosegment(file_path)
        new_data = _floatify_data(audiofile)

        reduced_noise_data = noisereduce.reduce_noise(
            new_data,
            noise_section,
            prop_decrease=prop_decrease,
            use_tensorflow=use_tensorflow,
            verbose=verbose,
            **kwargs,
        )

        reduced_noise_data = _int16ify_data(reduced_noise_data)
        audiofile._data = reduced_noise_data.astype(np.int16)

        file_name = os.path.basename(file_path)
        destination_name = os.path.join(destination_directory, file_name)
        if os.path.splitext(destination_name)[1].lower() in cant_write_ext:
            destination_name = os.path.splitext(destination_name)[0] + ".wav"

        print(f'Noise reduced for "{file_path}" writing to "{destination_name}"')

        with open(destination_name, "wb") as file_place:
            audiofile.export(
                file_place, format=os.path.splitext(destination_name)[1][1:]
            )

    except CouldntDecodeError:
        print(f"    Coudn't Decode {file_path}")


def shift_get_files(results: dict, sample_rate: int = None):
    names_and_paths = results.pop("names_and_paths")
    temp_a = results.pop("match_info")

    shifts_files = _shift_files(
        results,
        None,
        names_and_paths,
        None,
        sample_rate=sample_rate,
        return_files=True,
    )
    results["names_and_paths"] = names_and_paths
    results["match_info"] = temp_a
    return shifts_files


def shift_write_files(
    files_shifts: dict,
    destination_path: str,
    names_and_paths: dict,
    write_extension: str,
):
    _shift_files(
        files_shifts,
        destination_path,
        names_and_paths,
        write_extension,
        return_files=False,
    )


def _shift_files(
    files_shifts: dict,
    destination_path: str,
    names_and_paths: dict,
    write_extension: str,
    sample_rate: int = None,
    return_files: bool = False,
):
    if sample_rate is None:
        sample_rate = DEFAULT_FS
    max_shift = max(files_shifts.values())

    if write_extension:
        if write_extension[0] != ".":
            write_extension = "." + write_extension

    audsegs = {}
    for name in files_shifts.keys():
        file_path = names_and_paths[name]
        if return_files:
            audsegs[file_path] = max_shift - files_shifts[name]
        else:
            silence = AudioSegment.silent(
                (max_shift - files_shifts[name]) * 1000, frame_rate=sample_rate
            )

            audiofile = create_audiosegment(file_path, sample_rate=sample_rate)

            file_name = os.path.basename(file_path)
            audiofile: AudioSegment = silence + audiofile
            destination_name = os.path.join(destination_path, file_name)

            if os.path.splitext(destination_name)[1] in cant_write_ext:
                destination_name = os.path.splitext(destination_name)[0] + ".wav"

            if write_extension:
                destination_name = (
                    os.path.splitext(destination_name)[0] + write_extension
                )

                print(f"Writing {destination_name}")

                with open(destination_name, "wb") as file_place:
                    audiofile.export(
                        file_place, format=os.path.splitext(destination_name)[1][1:]
                    )

            else:
                print(f"Writing {destination_name}")

                with open(destination_name, "wb") as file_place:
                    audiofile.export(
                        file_place, format=os.path.splitext(destination_name)[1][1:]
                    )
            audsegs[file_path] = audiofile

    if return_files:
        return audsegs

    audsegs = list(audsegs.values())

    # adds silence to end of tracks to make them equally long for total
    longest_seconds = max(audseg.duration_seconds for audseg in audsegs)
    for i in range(len(audsegs)):
        audsegs[i] = audsegs[i] + AudioSegment.silent(
            (longest_seconds - audsegs[i].duration_seconds) * 1000,
            frame_rate=sample_rate,
        )

    # lower volume so the sum is the same volume
    total_files = audsegs[0] - (3 * math.log(len(files_shifts), 2))

    for i in audsegs[1:]:
        total_files = total_files.overlay(i - (3 * math.log(len(files_shifts), 2)))

    total_files = total_files.normalize()

    if write_extension:
        total_name = os.path.join(destination_path, "total") + write_extension
        print(f"Writing {total_name}")
        with open(total_name, "wb") as file_place:
            total_files.export(file_place, format=os.path.splitext(total_name)[1][1:])

    else:

        total_name = os.path.join(destination_path, "total.wav")

        print(f"Writing {total_name}")

        with open(total_name, "wb") as file_place:
            total_files.export(file_place, format=os.path.splitext(total_name)[1][1:])


def shift_write_file(file_path, destination_path, offset_seconds):

    silence = AudioSegment.silent(offset_seconds * 1000, frame_rate=DEFAULT_FS)

    audiofile = create_audiosegment(file_path)
    audiofile = silence + audiofile

    with open(destination_path, "wb") as file_place:
        audiofile.export(file_place, format=os.path.splitext(destination_path)[1][1:])


def get_shifted_file(file_path, offset_seconds, sample_rate=DEFAULT_FS) -> np.array:
    silence = AudioSegment.silent(offset_seconds * 1000, frame_rate=sample_rate)

    audiofile = create_audiosegment(file_path, sample_rate=sample_rate)
    audiofile = silence + audiofile
    return np.frombuffer(audiofile._data, np.int16)