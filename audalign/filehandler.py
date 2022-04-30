import fnmatch
import math
import multiprocessing
import os
import typing
from functools import partial

import noisereduce
import numpy as np
from numpy.core.defchararray import array
from pydub import AudioSegment, effects
from pydub.exceptions import CouldntDecodeError

from audalign.config import BaseConfig

cant_write_ext = [".mov", ".mp4", ".m4a"]
cant_read_ext = [".txt", ".md", ".pkf", ".py", ".pyc"]
can_read_ext = [
    ".mov",
    ".mp4",
    ".m4a",
    ".wav",
    ".WAV",
    ".mp3",
    ".MOV",
    ".ogg",
    ".aiff",
    ".aac",
    ".wma",
    ".flac",
]


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
                yield (p, os.path.splitext(p)[1])


def create_audiosegment(
    filepath: str,
    start_end: tuple = None,
    sample_rate=BaseConfig.sample_rate,
    length=None,
) -> AudioSegment:
    if sample_rate is None:
        sample_rate = BaseConfig.sample_rate
    if os.path.splitext(filepath)[1] in [".txt", ".json"]:
        raise CouldntDecodeError
    if len(filepath) > 0:
        audiofile = AudioSegment.from_file(filepath)
    else:
        if length is None:
            audiofile = AudioSegment.silent(duration=0, frame_rate=sample_rate)
        else:
            audiofile = AudioSegment.silent(duration=length, frame_rate=sample_rate)
    audiofile = audiofile.set_frame_rate(sample_rate)
    audiofile = audiofile.set_sample_width(2)
    audiofile = audiofile.set_channels(1)
    audiofile = effects.normalize(audiofile)
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


def get_audio_files_directory(directory_path: str, full_path: bool = False) -> list:
    """returns a list of the file paths in directory that are audio

    Args:
        directory_path (str): string of filepath

    Returns:
        list: of all paths in file that are audio
    """
    aud_list = []
    for file_path, ext in find_files(directory_path):
        if check_is_audio_file(file_path=file_path):
            if full_path is False:
                aud_list += [os.path.basename(file_path)]
            else:
                aud_list += [file_path]
    return aud_list


def check_is_audio_file(file_path: str) -> bool:
    ext = os.path.splitext(file_path)[1]
    try:
        if ext in [".txt", ".json"] or ext in cant_read_ext:
            return False
        elif ext.lower() not in can_read_ext:
            AudioSegment.from_file(file_path)
    except CouldntDecodeError:
        return False
    return True


def read(
    filename: str,
    wrdestination=None,
    start_end: tuple = None,
    sample_rate=BaseConfig.sample_rate,
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

    if os.path.splitext(filename)[1] in cant_read_ext:
        raise CouldntDecodeError
    audiofile = create_audiosegment(
        filename, start_end=start_end, sample_rate=sample_rate
    )
    data = np.frombuffer(audiofile._data, np.int16)
    if wrdestination:
        with open(wrdestination, "wb") as file_place:
            audiofile.export(file_place, format=os.path.splitext(wrdestination)[1][1:])
    return data, audiofile.frame_rate


def _floatify_data(audio_segment: AudioSegment):
    data = np.frombuffer(audio_segment._data, np.int16).astype(np.float32)
    data[np.where(data < 0)] /= 32768
    data[np.where(data > 0)] /= 32767
    return data


def _int16ify_data(data: array):
    data[np.where(data < 0)] *= 32768
    data[np.where(data > 0)] *= 32767
    return data.astype(np.int16)


def noise_remove(
    filepath,
    noise_start,
    noise_end,
    destination,
    write_extension: str = None,
    alt_noise_filepath=None,
    prop_decrease=1,
    **kwargs,
):
    audiofile = create_audiosegment(filepath)
    new_data = _floatify_data(audiofile)

    if not alt_noise_filepath:
        noisy_part = new_data[
            (noise_start * BaseConfig.sample_rate) : (
                noise_end * BaseConfig.sample_rate
            )
        ]
    else:
        noise_audiofile = create_audiosegment(alt_noise_filepath)
        noise_new_data = _floatify_data(noise_audiofile)
        noisy_part = noise_new_data[
            (noise_start * BaseConfig.sample_rate) : (
                noise_end * BaseConfig.sample_rate
            )
        ]

    print(f"Reducing noise: {filepath}")
    reduced_noise_data = noisereduce.reduce_noise(
        y=new_data,
        sr=BaseConfig.sample_rate,
        y_noise=noisy_part,
        prop_decrease=prop_decrease,
        **kwargs,
    )

    audiofile._data = _int16ify_data(reduced_noise_data)
    # if you pass in a folder for destination
    if len(os.path.splitext(destination)[1]) == 0:
        destination = os.path.join(destination, os.path.basename(filepath))
    if write_extension is not None:
        if write_extension[0] != ".":
            write_extension = "." + write_extension
        destination_name = os.path.splitext(destination)[0] + write_extension
        print(f"Writing {destination_name}")
        with open(destination_name, "wb") as file_place:
            audiofile.export(
                file_place, format=os.path.splitext(destination_name)[1][1:]
            )
    else:
        print(f"Writing {destination}")
        with open(destination, "wb") as file_place:
            audiofile.export(file_place, format=os.path.splitext(destination)[1][1:])


def noise_remove_directory(
    directory,
    noise_filepath,
    noise_start,
    noise_end,
    destination_directory,
    write_extension: str = None,
    prop_decrease=1,
    use_multiprocessing=False,
    num_processes=None,
    **kwargs,
):
    noise_data = _floatify_data(create_audiosegment(noise_filepath))[
        (noise_start * BaseConfig.sample_rate) : (noise_end * BaseConfig.sample_rate)
    ]
    file_names = []
    for file_path, _ in find_files(directory):
        file_names += [file_path]

    _reduce_noise = partial(
        _remove_noise,
        noise_section=noise_data,
        destination_directory=destination_directory,
        prop_decrease=prop_decrease,
        write_extension=write_extension,
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
    write_extension: str = None,
    destination_directory="",
    prop_decrease=1,
    **kwargs,
):

    try:
        print(f"Reducing noise: {file_path}")
        audiofile = create_audiosegment(file_path)
        new_data = _floatify_data(audiofile)

        reduced_noise_data = noisereduce.reduce_noise(
            y=new_data,
            sr=BaseConfig.sample_rate,
            y_noise=noise_section,
            prop_decrease=prop_decrease,
            **kwargs,
        )

        audiofile._data = _int16ify_data(reduced_noise_data)

        file_name = os.path.basename(file_path)
        destination_name = os.path.join(destination_directory, file_name)
        if os.path.splitext(destination_name)[1].lower() in cant_write_ext:
            destination_name = os.path.splitext(destination_name)[0] + ".wav"

        if write_extension is not None:
            if write_extension[0] != ".":
                write_extension = "." + write_extension
            destination_name = os.path.splitext(destination_name)[0] + write_extension
            print(f'Noise reduced for "{file_path}" writing to "{destination_name}"')
            with open(destination_name, "wb") as file_place:
                audiofile.export(
                    file_place, format=os.path.splitext(destination_name)[1][1:]
                )
        else:
            print(f'Noise reduced for "{file_path}" writing to "{destination_name}"')
            with open(destination_name, "wb") as file_place:
                audiofile.export(
                    file_place, format=os.path.splitext(destination_name)[1][1:]
                )

    except CouldntDecodeError:
        print(f"    Coudn't Decode {file_path}")


def calc_array_indexes(array_length, width, overlap_ratio):
    index_list = []
    if width > array_length:
        index_list += [0]
    else:
        [
            index_list.append(i)
            for i in range(
                0, array_length - int(width), int(width * (1 - overlap_ratio))
            )
        ]
        if (
            array_length - int(width) not in index_list
            and array_length - int(width) > 0
        ):
            index_list.append(array_length - int(width))
    return index_list


def calc_overlap_array(length, index_list, width):
    overlap_array = np.zeros(length, dtype=np.float32)
    for index in index_list:
        overlap_array[index : index + width] += 1
    return overlap_array


def uniform_level_directory(
    directory: str,
    destination: str,
    write_extension: str = None,
    mode: str = "normalize",
    width: float = 5,
    overlap_ratio=0.5,
    exclude_min_db=-70,
    use_multiprocessing=False,
    num_processes=None,
):
    _uniform_level_ = partial(
        _uniform_level,
        destination_name=destination,
        write_extension=write_extension,
        mode=mode,
        width=width,
        overlap_ratio=overlap_ratio,
        exclude_min_db=exclude_min_db,
    )

    if use_multiprocessing == True:

        try:
            nprocesses = num_processes or multiprocessing.cpu_count()
        except NotImplementedError:
            nprocesses = 1
        else:
            nprocesses = 1 if nprocesses <= 0 else nprocesses

        with multiprocessing.Pool(nprocesses) as pool:

            pool.map(_uniform_level_, (x[0] for x in find_files(directory)))

            pool.close()
            pool.join()
    else:
        for i in (x[0] for x in find_files(directory)):
            _uniform_level_(i)


def _uniform_level(
    file_path: str,
    destination_name: str,
    write_extension: str = None,
    mode: str = "normalize",
    width: float = 5,
    overlap_ratio=0.5,
    exclude_min_db=-70,
):
    assert overlap_ratio < 1 and overlap_ratio >= 0
    try:
        print(f"Uniform Leveling: {file_path}")
        audiofile = create_audiosegment(file_path)
        audiofile_data = np.frombuffer(audiofile._data, np.int16)
        width *= BaseConfig.sample_rate
        if width > len(audiofile_data):
            width = len(audiofile_data)
        index_list = calc_array_indexes(
            len(audiofile_data),
            width,
            overlap_ratio,
        )
        overlap_array = calc_overlap_array(len(audiofile_data), index_list, width)
        if mode == "normalize":
            audiofile._data = level_by_normalize(
                audiofile_data,
                index_list,
                overlap_array,
                width,
                exclude_min_db,
            ).astype(np.int16)
        elif mode == "average":
            audiofile._data = level_by_ave(
                audiofile_data,
                index_list,
                overlap_array,
                width,
                exclude_min_db,
            ).astype(np.int16)
        else:
            raise ValueError(
                f'Mode must be either "normalize" or "average", not {mode}'
            )
        audiofile = effects.normalize(audiofile)

        file_name = os.path.basename(file_path)
        if len(os.path.splitext(destination_name)[1]) == 0:
            destination_name = os.path.join(destination_name, file_name)
        if os.path.splitext(destination_name)[1].lower() in cant_write_ext:
            destination_name = os.path.splitext(destination_name)[0] + ".wav"

        if write_extension is not None:
            if write_extension[0] != ".":
                write_extension = "." + write_extension
            destination_name = os.path.splitext(destination_name)[0] + write_extension
            print(f'Uniform leveled "{file_path}" writing to "{destination_name}"')
            with open(destination_name, "wb") as file_place:
                audiofile.export(
                    file_place, format=os.path.splitext(destination_name)[1][1:]
                )
        else:
            print(f'Uniform leveled "{file_path}" writing to "{destination_name}"')
            with open(destination_name, "wb") as file_place:
                audiofile.export(
                    file_place, format=os.path.splitext(destination_name)[1][1:]
                )
    except CouldntDecodeError:
        print(f"    Coudn't Decode {file_path}")


def level_by_normalize(
    audiofile_data, index_list, overlap_array, width, exclude_min_db
):
    new_audio_data = np.zeros(len(audiofile_data), dtype=np.float32)
    silent_audiosegment = create_audiosegment(
        "", length=width / BaseConfig.sample_rate * 1000
    )
    for index in index_list:
        silent_audiosegment._data = audiofile_data[index : index + width]
        if silent_audiosegment.max_dBFS < exclude_min_db:
            continue
        silent_audiosegment = effects.normalize(silent_audiosegment)
        new_audio_data[index : index + width] += np.frombuffer(
            silent_audiosegment._data, dtype=np.int16
        )
    new_audio_data /= overlap_array
    return new_audio_data


def level_by_ave(audiofile_data, index_list, overlap_array, width, exclude_min_db):
    new_audio_data = np.zeros(len(audiofile_data), dtype=np.float32)
    audiosegment_slicer = create_audiosegment(
        "", length=width / BaseConfig.sample_rate * 1000
    )
    average_level_list = []

    for index in index_list:
        audiosegment_slicer._data = audiofile_data[index : index + width]
        if audiosegment_slicer.max_dBFS < exclude_min_db:
            continue
        average_level_list.append(
            (index, audiosegment_slicer.dBFS, audiosegment_slicer.max_dBFS)
        )
    target_ave = max(average_level_list, key=lambda x: x[2] - x[1])
    target_ave = target_ave[2] - target_ave[1]
    for index, ave_db, _ in average_level_list:
        audiosegment_slicer._data = audiofile_data[index : index + width]
        audiosegment_slicer = audiosegment_slicer + (-ave_db - target_ave)
        new_audio_data[index : index + width] += np.frombuffer(
            audiosegment_slicer._data, dtype=np.int16
        )
    new_audio_data /= overlap_array
    return new_audio_data


def shift_get_files(results: dict, sample_rate: int = None):
    names_and_paths = results.pop("names_and_paths")
    temp_a = results.pop("match_info")
    temp_rankings = None
    if results.get("rankings") is not None:
        temp_rankings = results.pop("rankings")

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
    if temp_rankings is not None:
        results["rankings"] = temp_rankings
    return shifts_files


def shift_write_files(
    files_shifts: dict,
    destination_path: str,
    names_and_paths: dict,
    write_extension: str,
    write_multi_channel: bool = False,
):
    """
    Args
    ----
        files_shifts (dict{float}): dict with file path as key and shift as value
        destination_path (str): folder to write file to
        names_and_paths (dict{str}): dict with name as key and path as value
        write_extension (str): if given, writes all alignments with given extension (ex. ".wav" or "wav")
        write_multi_channel (bool): If true, only write out combined file with each input audio file being one channel. If false, write out shifted files separately and total combined file
    """
    _shift_files(
        files_shifts,
        destination_path,
        names_and_paths,
        write_extension,
        write_multi_channel=write_multi_channel,
        return_files=False,
    )


def _shift_files(
    files_shifts: dict,
    destination_path: typing.Optional[str],
    names_and_paths: dict,
    write_extension: typing.Optional[str],
    write_multi_channel: bool = False,
    sample_rate: int = None,
    return_files: bool = False,
):
    if sample_rate is None:
        sample_rate = BaseConfig.sample_rate

    if write_extension:
        if write_extension[0] != ".":
            write_extension = "." + write_extension

    if not write_multi_channel:
        return _shift_write_separate(
            files_shifts,
            destination_path,
            names_and_paths,
            write_extension,
            sample_rate=sample_rate,
            return_files=return_files,
        )
    else:
        return _shift_write_multichannel(
            files_shifts,
            destination_path,
            names_and_paths,
            write_extension,
            sample_rate=sample_rate,
            return_files=return_files,
        )


def _shift_write_separate(
    files_shifts: dict,
    destination_path: typing.Optional[str],
    names_and_paths: dict,
    write_extension: typing.Optional[str],
    sample_rate: int = None,
    return_files: bool = False,
):
    audsegs = _shift_prepend_space_audsegs(
        files_shifts=files_shifts,
        names_and_paths=names_and_paths,
        sample_rate=sample_rate,
        return_files=return_files,
    )
    if return_files:
        return audsegs
    for file_path, audseg in audsegs.items():
        _write_single_shift(
            audiofile=audseg,
            file_path=file_path,
            destination_path=destination_path,
            write_extension=write_extension,
        )

    audsegs = list(audsegs.values())

    # adds silence to end of tracks to make them equally long for total
    longest_bytes = max(len(audseg._data) for audseg in audsegs)
    for i in range(len(audsegs)):
        data = audsegs[i]._data
        len_difference = longest_bytes - len(data)
        if len_difference > 0:
            new_zeros = bytearray(len_difference)
            audsegs[i]._data = data + new_zeros

    # lower volume so the sum is the same volume
    total_files = audsegs[0] - (3 * math.log(len(files_shifts), 2))

    for i in audsegs[1:]:
        total_files = total_files.overlay(i - (3 * math.log(len(files_shifts), 2)))

    total_files = effects.normalize(total_files)

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


def _shift_prepend_space_audsegs(
    files_shifts: dict,
    names_and_paths: dict,
    sample_rate: int,
    return_files: bool = False,
):
    audsegs = {}
    for name in files_shifts.keys():
        file_path = names_and_paths[name]
        if return_files:
            audsegs[file_path] = files_shifts[name]
        else:
            silence = AudioSegment.silent(
                (files_shifts[name]) * 1000, frame_rate=sample_rate
            )
            audiofile = create_audiosegment(file_path, sample_rate=sample_rate)
            audiofile: AudioSegment = silence + audiofile
            audsegs[file_path] = audiofile
    return audsegs


def _write_single_shift(
    audiofile: AudioSegment,
    file_path: str,
    destination_path: str,
    write_extension: str,
):

    file_name = os.path.basename(file_path)
    destination_name = os.path.join(destination_path, file_name)  # type: ignore

    if os.path.splitext(destination_name)[1] in cant_write_ext:
        destination_name = os.path.splitext(destination_name)[0] + ".wav"

    if write_extension:
        destination_name = os.path.splitext(destination_name)[0] + write_extension

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


def _shift_write_multichannel(
    files_shifts: dict,
    destination_path: typing.Optional[str],
    names_and_paths: dict,
    write_extension: typing.Optional[str],
    sample_rate: int,
    return_files: bool = False,
):
    audsegs = _shift_prepend_space_audsegs(
        files_shifts=files_shifts,
        names_and_paths=names_and_paths,
        sample_rate=sample_rate,
        return_files=return_files,
    )
    if return_files:
        return audsegs
    # sorts channels by filename
    audsegs = [x[1] for x in sorted(list(audsegs.items()))]
    # adds silence to end of tracks to make them equally long for total
    longest_bytes = max(len(audseg._data) for audseg in audsegs)
    for i in range(len(audsegs)):
        data = audsegs[i]._data
        len_difference = longest_bytes - len(data)
        if len_difference > 0:
            new_zeros = bytearray(len_difference)
            audsegs[i]._data = data + new_zeros
    total_files = AudioSegment.from_mono_audiosegments(*audsegs)
    total_files = effects.normalize(total_files)

    if write_extension:
        total_name = os.path.join(destination_path, "multi_channel_total") + write_extension  # type: ignore
        print(f"Writing {total_name}")
        with open(total_name, "wb") as file_place:
            total_files.export(file_place, format=os.path.splitext(total_name)[1][1:])
    else:
        total_name = os.path.join(destination_path, "multi_channel_total.wav")  # type: ignore
        print(f"Writing {total_name}")
        with open(total_name, "wb") as file_place:
            total_files.export(file_place, format=os.path.splitext(total_name)[1][1:])


def shift_write_file(file_path, destination_path, offset_seconds):
    silence = AudioSegment.silent(
        offset_seconds * 1000, frame_rate=BaseConfig.sample_rate
    )
    audiofile = create_audiosegment(file_path)
    audiofile = silence + audiofile
    with open(destination_path, "wb") as file_place:
        audiofile.export(file_place, format=os.path.splitext(destination_path)[1][1:])


def get_shifted_file(
    file_path, offset_seconds, sample_rate=BaseConfig.sample_rate
) -> np.array:
    silence = AudioSegment.silent(offset_seconds * 1000, frame_rate=sample_rate)

    audiofile = create_audiosegment(file_path, sample_rate=sample_rate)
    audiofile = silence + audiofile
    return np.frombuffer(audiofile._data, np.int16)
