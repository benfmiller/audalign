"""
This file serves as the host file for all audalign functions.

recognitions and alignments use recognizer objects which can be configured with
their respective configuration objects. 

There are a number of functions that can be used to enhance the alignments such
as "uniform leveling" and "remove noise" functions. Writing shifts can also be nifty.

One useful workflow could be to uniform level some files, do a remove noise on them, then
align and fine align them. Then, to have the shifts applied to the original files, use
"write_shifts_from_results".
"""

import os
from functools import wraps
from pprint import PrettyPrinter

from pydub.utils import mediainfo

import audalign.align as aligner
import audalign.datalign as datalign
import audalign.filehandler as filehandler
from audalign.config import BaseConfig
from audalign.recognizers import BaseRecognizer
from audalign.recognizers.correcognize import CorrelationRecognizer
from audalign.recognizers.correcognizeSpectrogram import \
    CorrelationSpectrogramRecognizer
from audalign.recognizers.fingerprint import FingerprintRecognizer
from audalign.recognizers.visrecognize import VisualRecognizer


def add_rankings(func):
    @wraps(func)
    def wrapper_decorator(*args, **kwargs):
        results = func(*args, **kwargs)
        if results is not None:
            if results.get("rankings") is not None:
                results["rankings"]["fine_match_info"] = rank_alignment(
                    results, kwargs["recognizer"]
                )
            else:
                results["rankings"] = {}
                recognizer = kwargs.get("recognizer")
                if recognizer is None:
                    if results.get("fine_match_info") is None:
                        recognizer = FingerprintRecognizer()
                    else:
                        recognizer = CorrelationRecognizer()
                results["rankings"]["match_info"] = rank_alignment(results, recognizer)

        return results

    return wrapper_decorator


@add_rankings
def recognize(
    target_file: str,
    against_path: str = None,
    recognizer: BaseRecognizer = None,
) -> dict:
    if recognizer is None:
        recognizer = FingerprintRecognizer()
    return recognizer.recognize(target_file, against_path)


@add_rankings
def align(
    directory_path: str,
    destination_path: str = None,
    write_extension: str = None,
    write_multi_channel: bool = False,
    recognizer: BaseRecognizer = None,
):
    """
    Finds matches and relative offsets for all files in directory_path, aligns them, and writes them to destination_path

    Args
    ----
        directory_path (str): String of directory for alignment
        destination_path (str): String of path to write alignments to
        write_extension (str): if given, writes all alignments with given extension (ex. ".wav" or "wav")
        recognizer (BaseRecognizer, optional): recognizer object
        write_multi_channel (bool): If true, only write out combined file with each input audio file being one channel. If false, write out shifted files separately and total combined file

    Returns
    -------
        files_shifts (dict{float}): dict of file name with shift as value
    """
    if recognizer is None:
        recognizer = FingerprintRecognizer()
    return aligner._align(
        recognizer=recognizer,
        filename_list=None,
        file_dir=directory_path,
        destination_path=destination_path,
        write_extension=write_extension,
        write_multi_channel=write_multi_channel,
    )


@add_rankings
def align_files(
    filename_a,
    filename_b,
    *filenames,
    destination_path: str = None,
    write_extension: str = None,
    write_multi_channel: bool = False,
    recognizer: BaseRecognizer = None,
):
    """
    Finds matches and relative offsets for all files given, aligns them, and writes them to destination_path if given

    try align_files(*filepath_list) for more concision if desired

    Args
    ----
        filename_a (str): String of path for alignment
        filename_b (str): String of path for alignment
        *filenames (strs): strings of paths for alignment
        destination_path (str): String of path to write alignments to
        write_extension (str): if given, writes all alignments with given extension (ex. ".wav" or "wav")
        write_multi_channel (bool): If true, only write out combined file with each input audio file being one channel. If false, write out shifted files separately and total combined file
        recognizer (BaseRecognizer, optional): recognizer object

    Returns
    -------
        files_shifts (dict{float}): dict of file name with shift as value
    """
    filename_list = [filename_a, filename_b, *filenames]
    if recognizer is None:
        recognizer = FingerprintRecognizer()
    return aligner._align(
        recognizer=recognizer,
        filename_list=filename_list,
        file_dir=None,
        destination_path=destination_path,
        write_extension=write_extension,
        write_multi_channel=write_multi_channel,
    )


@add_rankings
def target_align(
    target_file: str,
    directory_path: str,
    destination_path: str = None,
    write_extension: str = None,
    write_multi_channel: bool = False,
    recognizer: BaseRecognizer = None,
):
    """matches and relative offsets for all files in directory_path using only target file,
    aligns them, and writes them to destination_path if given. Uses fingerprinting by defualt,
    but uses visual recognition if false

    Args
    ----
        target_file (str): File to find alignments against
        directory_path (str): Directory to align against
        destination_path (str, optional): Directory to write alignments to
        write_extension (str, optional): audio file format to write to. Defaults to None.
        write_multi_channel (bool): If true, only write out combined file with each input audio file being one channel. If false, write out shifted files separately and total combined file
        recognizer (BaseRecognizer, optional): recognizer object

    Returns
    -------
        dict: dict of file name with shift as value along with match info
    """

    if recognizer is None:
        recognizer = FingerprintRecognizer()
    return aligner._align(
        recognizer=recognizer,
        filename_list=[target_file],
        file_dir=directory_path,
        destination_path=destination_path,
        target_aligning=True,
        write_extension=write_extension,
        write_multi_channel=write_multi_channel,
    )


@add_rankings
def fine_align(
    results,
    destination_path: str = None,
    write_extension: str = None,
    write_multi_channel: bool = False,
    match_index: int = 0,
    recognizer: BaseRecognizer = None,
):
    """
    Finds matches and relative offsets for all files in directory_path, aligns them, and writes them to destination_path

    Args
    ----
        results (dict): results from previous alignments.
        destination_path (str): String of path to write alignments to
        write_extension (str): if given, writes all alignments with given extension (ex. ".wav" or "wav")
        write_multi_channel (bool): If true, only write out combined file with each input audio file being one channel. If false, write out shifted files separately and total combined file
        match_index (int): reorders the input results to the given match index.
        recognizer (BaseRecognizer, optional): recognizer object

    Returns
    -------
        files_shifts (dict{float}): dict of file name with shift as value. Includes match_info, fine_match_info, and names_and_paths
    """

    if results is None:
        raise ValueError(f'results "{results}", no results to fine align')

    print("Fine Aligning...")

    if recognizer is None:
        recognizer = CorrelationRecognizer()

    if match_index != 0:
        recalc_shifts_results = aligner.recalc_shifts_index(
            results,
            key="match_info",
            match_index=match_index,
            strength_stat=recognizer.config.CONFIDENCE,
        )
        paths_audio = filehandler.shift_get_files(
            recalc_shifts_results, sample_rate=recognizer.config.sample_rate
        )
    else:
        paths_audio = filehandler.shift_get_files(
            results, sample_rate=recognizer.config.sample_rate
        )

    max_lags_not_set = False
    if recognizer.config.max_lags is None:
        recognizer.config.max_lags = 2
        max_lags_set = True
    new_results = aligner._align(
        recognizer=recognizer,
        filename_list=None,
        file_dir=None,
        fine_aud_file_dict=paths_audio,
    )
    if max_lags_not_set is True:
        recognizer.config.max_lags = None
    if new_results is None:
        print("No matches found for fine alignment")
        return
    new_results = aligner.combine_fine(results, new_results)

    if destination_path:
        copy_dict = {}
        for name, value in new_results.items():
            if name not in [
                "names_and_paths",
                "match_info",
                "fine_match_info",
                "rankings",
            ]:
                copy_dict[name] = value
        try:
            _write_shifted_files(
                copy_dict,
                destination_path,
                new_results["names_and_paths"],
                write_extension,
                write_multi_channel=write_multi_channel,
            )
        except PermissionError:
            print("Permission Denied for write fine_align")

    return new_results


# --------------------------------------------------------------------------------------


def write_processed_file(
    file_path: str,
    destination_file: str,
    start_end: tuple = None,
    sample_rate: int = BaseConfig.sample_rate,
) -> None:
    """
    writes given file to the destination file after processing for fingerprinting

    Args
    ----
        file_path (str): file path of audio file
        destination_file (str): file path and name to write file to
        start_end (tuple(float, float), optional): Silences before and after start and end. (0, -1) Silences last second, (5.4, 0) silences first 5.4 seconds
        sample_rate (int): sample rate to write file to

    """
    filehandler.read(
        filename=file_path,
        wrdestination=destination_file,
        start_end=start_end,
        sample_rate=sample_rate,
    )


def plot_peaks(
    file_path: str,
    config: BaseConfig = None,
) -> None:
    """
    Plots the file_path's peak chart from fingerprinting

    Args
    ----
        file_path (str): file to plot
        start_end (tuple(float, float), optional): Silences before and after start and end. (0, -1) Silences last second, (5.4, 0) silences first 5.4 seconds

    Returns
    -------
    None
    """
    fing_rec = FingerprintRecognizer(config)
    fing_rec.config.plot = True
    fing_rec._fingerprint_file(
        file_path,
    )


def rank_alignment(
    alignment: dict,
    recognizer: BaseRecognizer,
):
    """
    Ranks alignments and recognitions.
    Included as "rankings" in every recognition and alignment
    Ranks each recognition on a scale of 1-10
    Ranks are not proof of a good alignment, just a tool to gauge the different confidences and techniques

    Args
    ----
        alignment (dict): result from either recognition or alignment

    Returns
    -------
        dict: form similar to alignments or matches

    """
    return datalign.rank_alignment(alignment=alignment, recognizer=recognizer)


def pretty_print_results(results, match_keys="both"):
    """
    nifty printer for recognition or alignment results
    match_keys goes to pretty_print_alignment

    Args
    ----
        results (dict): result from recognition or alignment
        match_keys (str): "both", "match_info", or "fine_match_info"

    Returns
    -------
        None
    """
    if results:
        if "names_and_paths" not in results:
            pretty_print_recognition(results)
        else:
            pretty_print_alignment(results, match_keys=match_keys)
    else:
        print("No Matches Found")


def pretty_print_recognition(results, _in_alignment: bool = False):
    """
    nifty printer for recognition results

    Args
    ----
        results (dict): result from recognition

    Returns
    -------
        None
    """
    max_conf = 0
    min_conf = 999999999999
    if results:
        strength_stat = "confidence"
        _ = results["match_info"]
        _ = _[list(_.keys())[0]]
        if "confidence" not in _:
            strength_stat = "ssim"
        elif "scaling_factor" in _:
            strength_stat = "scaling_factor"
        audio_file_list = sorted(
            list(results["match_info"].keys()),
            key=lambda x: _recog_sorter(x, results, strength_stat),
        )
        max_conf = max(
            list(results["match_info"].keys()),
            key=lambda x: _recog_sorter(x, results, strength_stat),
        )
        max_conf = _recog_sorter(max_conf, results, strength_stat)
        min_conf = min(
            list(results["match_info"].keys()),
            key=lambda x: _recog_sorter(x, results, strength_stat),
        )
        min_conf = _recog_sorter(min_conf, results, strength_stat)
        for audio_file in audio_file_list:
            if not _in_alignment:
                print(f"\n{audio_file}")
            else:
                print(f"\nmatched with {audio_file}")
            print(f"Match time: {seconds_to_min_hrs(results.get('match_time'))}")
            for section, info in results["match_info"][audio_file].items():
                if type(info) == type([]):
                    if section in [
                        "locality_frames",
                        "offset_frames",
                        "locality_samples",
                        "offset_samples",
                    ]:
                        continue
                    elif type(info) == type([]):
                        if section == "mse":
                            if info[0] == 20000000:
                                continue
                        elif section == "locality_seconds":
                            print(f"{section}: [", end="")
                            for num, i in enumerate(info):
                                if num > 9:
                                    break
                                if i is not None:
                                    print(f"{i[0]} {len(i)} : ", end="")
                                else:
                                    print("None : ", end="")
                            print("\b\b\b] ")
                            continue
                        if info is None or info[0] is None:
                            continue
                        print(f"{section}: {info[0:10] if len(info) > 9 else info}")
                        print(f'max "{section}" is {max(info)}: min is {min(info)}')
                else:
                    print(f"{section}, {info}")
        if not _in_alignment:
            if results.get("rankings") is not None:
                print()
                PrettyPrinter().pprint(results["rankings"])
            print()
            for name, result in results["match_info"].items():
                print(f"{name} : {result['offset_seconds'][0]}")
    else:
        print("No Matches Found")
    return min_conf, max_conf


def _recog_sorter(x, result, strength_stat):
    if strength_stat == "scaling_factor":
        return result["match_info"][x][strength_stat]
    return result["match_info"][x][strength_stat][0]


def pretty_print_alignment(results, match_keys="both"):
    """
    nifty printer for alignment results
    if match_keys is both, prints "match_info" and "fine_match_info" if present

    Args
    ----
        results (dict): result from alignment
        match_keys (str): "both", "match_info", or "fine_match_info"

    Returns
    -------
        None
    """
    min_conf_list, max_conf_list = [], []
    if results:
        if match_keys == "both":
            match_keys = ["match_info"]
            if results.get("fine_match_info") is not None:
                match_keys += ["fine_match_info"]
        else:
            match_keys = [match_keys]
        for match_key in match_keys:
            print(f'\n        Key: "{match_key}"')
            for audio_file in results.get(match_key).keys():
                print(f"\n    {audio_file}")
                min_conf, max_conf = pretty_print_recognition(
                    results[match_key][audio_file], _in_alignment=True
                )
                min_conf_list += [min_conf]
                max_conf_list += [max_conf]
        if results.get("rankings") is not None:
            print("\nRankings")
            PrettyPrinter().pprint(results["rankings"])
        print()
        for match in results.keys():
            if match not in [
                "match_info",
                "fine_match_info",
                "names_and_paths",
                "rankings",
            ]:
                print(f"{match} : {results[match]}")
        print()
        print(f"max conf is {max(max_conf_list)} : min conf is {min(min_conf_list)}")
    else:
        print("No Matches Found")
    print()


def recalc_shifts(
    results: dict,
    key: str = None,
    match_index: int = 0,
    fine_match_index: int = 0,
    strength_stat: str = None,
    fine_strength_stat: str = None,
):
    """
    Takes results from alignment and recalculates alignment

    key is either "match_info", "fine_match_info", or "only_fine_match_info"
    "match_info" is like undoing the fine alignment. "fine_match_info" is like redoing the fine alignment.
    "only_fine_match_info" gives the relative shifts only calculated by fine alignment without the first alignment.

    Args
    ----
        results (dict): results from alignment.
        key (int, optional): key to recalculate from. defaults to "fine_match_info" if present.
        match_index (int, optional): index of match to reorder by. First index (0) is usually the strongest match.
        fine_match_index (int, optional): index of match to reorder by for fine_match_info. First index (0) is usually the strongest match.
        strength_stat (str, optional): change if using mse for visual *not recommended*.
        fine_strength_stat (str, optional): change if using mse for visual *not recommended*.
    """
    return aligner.recalc_shifts_index(
        results,
        key=key,
        match_index=match_index,
        fine_match_index=fine_match_index,
        strength_stat=strength_stat,
        fine_strength_stat=fine_strength_stat,
    )


def _write_shifted_files(
    files_shifts: dict,
    destination_path: str,
    names_and_paths: dict,
    write_extension: str,
    write_multi_channel: bool = False,
):
    """
    Writes files to destination_path with specified shift

    Args
    ----
        files_shifts (dict{float}): dict with file path as key and shift as value
        destination_path (str): folder to write file to
        names_and_paths (dict{str}): dict with name as key and path as value
        write_multi_channel (bool): If true, only write out combined file with each input audio file being one channel. If false, write out shifted files separately and total combined file
    """
    filehandler.shift_write_files(
        files_shifts,
        destination_path,
        names_and_paths,
        write_extension,
        write_multi_channel=write_multi_channel,
    )


def get_metadata(file_path: str):
    """Returns metadata of audio or video file

    if file_path is not a valid file or is a directory, returns empty dict

    Args
    ----
        file_path (str): file path to file

    Returns
    -------
        dict: dict of tags and values
    """
    return mediainfo(filepath=file_path)


def write_shifted_file(file_path: str, destination_path: str, offset_seconds: float):
    """
    Writes file to destination_path with specified shift in seconds

    Args
    ----
        file_path (str): file path of file to shift
        destination_path (str): where to write file to and file name
        offset_seconds (float): how many seconds to shift, can't be negative
    """
    filehandler.shift_write_file(file_path, destination_path, offset_seconds)


def write_shifts_from_results(
    results: dict,
    read_from_dir,
    destination_path: str,
    write_extension: str = None,
    write_multi_channel: bool = False,
):
    """
    For writing the results of an alignment with alternate source files.
    Especially useful if you want to align the original files with the results from noise
    removed or uniform leveled versions.

    Read_from_dir can be a str of directory, a list of file path str's, or None to use original source files.

    Cannot have two files with same basename ("Test.wav" and "Test.mp3" in same read_from_dir is undefined)

    Args
    ----
        results (dict): results from alignment.
        read_from_dir (str, list): source files for writing. or None to use original files.
        destination_path (str): destination to write to.
        write_extension (str, optional): if given, all files writen with given extension
        write_multi_channel (bool): If true, only write out combined file with each input audio file being one channel. If false, write out shifted files separately and total combined file
    """
    if isinstance(read_from_dir, str):
        print("Finding audio files")
        read_from_dir = filehandler.get_audio_files_directory(
            read_from_dir, full_path=True
        )
    if read_from_dir is not None:
        results_files = {}
        for path in results.keys():
            if len(os.path.splitext(os.path.basename(path))[1]) > 0:
                results_files[os.path.splitext(os.path.basename(path))[0]] = path

        copy_dict = {}
        names_and_paths = {}
        for filename in read_from_dir:
            base_basename = os.path.splitext(os.path.basename(filename))[0]
            if base_basename in results_files:
                copy_dict[base_basename] = results[results_files[base_basename]]
                names_and_paths[base_basename] = filename
        if copy_dict == {}:
            print("No matching file basenames found")
            return
    else:
        copy_dict = {}
        for name, value in results.items():
            if name not in [
                "names_and_paths",
                "match_info",
                "fine_match_info",
                "rankings",
            ]:
                copy_dict[name] = value
        names_and_paths = results["names_and_paths"]
    try:
        _write_shifted_files(
            copy_dict,
            destination_path,
            names_and_paths,
            write_extension,
            write_multi_channel=write_multi_channel,
        )
    except PermissionError:
        print("Permission Denied for write fine_align")


def convert_audio_file(
    file_path: str,
    destination_path: str,
    start_end: tuple = None,
    sample_rate: int = None,
):
    """
    Convert audio file to type specified in destination path

    Args
    ----
        file_path (str): file path of file to shift
        destination_path (str): where to write file to and file name
        start_end (tuple(float, float), optional): Silences before and after start and end. (0, -1) Silences last second, (5.4, 0) silences first 5.4 seconds
        sample_rate (int): sample rate to write file to
    """
    filehandler.read(
        filename=file_path,
        wrdestination=destination_path,
        start_end=start_end,
        sample_rate=sample_rate,
    )


# -----------------------------------------------------------------------------------------


def uniform_level_file(
    file_path: str,
    destination: str,
    write_extension: str = None,
    mode: str = "normalize",
    width: float = 5,
    overlap_ratio: float = 0.5,
    exclude_min_db: float = -70,
) -> None:
    """
    Levels the file using either of two methods: normalize or average.
    "normalize" levels the file by peak volume while "average" levels by average volume.

    This function pairs well with remove_noise. Leveling brings sound events up in volume
    so they influence alignments. Removing noise cancels noise that might have been boosted
    by the leveling.

    Args
    ----
        file_path (str): path of file to level.
        destination (str): path of destination to write to. directory or file name.
        write_extension (str, optional): extension/format for writing.
        mode (str): either "normalize" or "average".
        width (float): width in seconds for each leveling.
        overlap_ratio (float): between 1 and 0. overlapping windows.
        exclude_min_db (float): less than 0. Doesn't level window with max dBFS lower than this.

    Returns
    -------
        None
    """
    filehandler._uniform_level(
        file_path=file_path,
        destination_name=destination,
        write_extension=write_extension,
        mode=mode,
        width=width,
        overlap_ratio=overlap_ratio,
        exclude_min_db=exclude_min_db,
    )


def uniform_level_directory(
    directory: str,
    destination: str,
    write_extension: str = None,
    mode: str = "normalize",
    width: float = 5,
    overlap_ratio: float = 0.5,
    exclude_min_db: float = -70,
    multiprocessing: bool = True,
    num_processors: int = None,
) -> None:
    """
    Levels the file using either of two methods: normalize or average.
    "normalize" levels the file by peak volume while "average" levels by average volume.

    This function pairs well with remove_noise. Leveling brings sound events up in volume
    so they influence alignments. Removing noise cancels noise that might have been boosted
    by the leveling.

    Args
    ----
        file_path (str): path of file to level.
        destination (str): path of destination to write to. directory or file name.
        write_extension (str, optional): extension/format for writing.
        mode (str): either "normalize" or "average".
        width (float): width in seconds for each leveling.
        overlap_ratio (float): between 1 and 0. overlapping windows.
        exclude_min_db (float): less than 0. Doesn't level window with max dBFS lower than this.
        multiprocessing (bool): If true, uses multiprocessing
        num_processors (int, optional): number of processors to use

    Returns
    -------
        None
    """
    filehandler.uniform_level_directory(
        directory=directory,
        destination=destination,
        write_extension=write_extension,
        mode=mode,
        width=width,
        overlap_ratio=overlap_ratio,
        exclude_min_db=exclude_min_db,
        use_multiprocessing=multiprocessing,
        num_processes=num_processors,
    )


def remove_noise_file(
    filepath: str,
    noise_start: float,
    noise_end: float,
    destination: str,
    write_extension: str = None,
    alt_noise_filepath: str = None,
    prop_decrease: float = 1,
    **kwargs,
):
    """Remove noise from audio file by specifying start and end seconds of representative sound sections. Writes file to destination

    Args
    ----
        filepath (str): filepath to read audio file
        noise_start (float): positition in seconds of start of noise section
        noise_end (float): position in seconds of end of noise section
        destination (str): filepath of destination to write to, full path or directory
        write_extension (str): if given, writes all alignments with given extension (ex. ".wav" or "wav")
        alt_noise_filepath (str): path of different file for noise sample
        prop_decrease (float): between 0 and 1. Proportion to decrease noise
        kwargs : kwargs for noise reduce. Look at noisereduce kwargs in filehandler
    """
    filehandler.noise_remove(
        filepath,
        noise_start,
        noise_end,
        destination,
        write_extension=write_extension,
        alt_noise_filepath=alt_noise_filepath,
        prop_decrease=prop_decrease,
        **kwargs,
    )


def remove_noise_directory(
    directory: str,
    noise_filepath: str,
    noise_start: float,
    noise_end: float,
    destination_directory: str,
    write_extension: str = None,
    prop_decrease: float = 1,
    multiprocessing: bool = True,
    num_processors: int = None,
    **kwargs,
):
    """Remove noise from audio files in directory by specifying start and end seconds of
    representative sound sections.  Writes file to destination directory

    Args
    ----
        directory (str): filepath for directory to quiet
        nosie_filepath (str): filepath to read noise file
        noise_start (float): positition in seconds of start of noise section
        noise_end (float): position in seconds of end of noise section
        destination_directory (str): filepath of destination directory to write to
        write_extension (str): if given, writes all alignments with given extension (ex. ".wav" or "wav")
        prop_decrease (float): between 0 and 1. Proportion to decrease noise
        multiprocessing (bool): If true, uses multiprocessing
        num_processors (int, optional): number of processors to use
        kwargs : kwargs for noise reduce. Look at noisereduce kwargs in filehandler
    """
    filehandler.noise_remove_directory(
        directory,
        noise_filepath,
        noise_start,
        noise_end,
        destination_directory,
        write_extension=write_extension,
        prop_decrease=prop_decrease,
        use_multiprocessing=multiprocessing,
        num_processes=num_processors,
        **kwargs,
    )


# -----------------------------------------------------------------------------------------


def seconds_to_min_hrs(seconds):
    if seconds > 60:
        minutes = seconds // 60
        seconds = seconds % 60
        if minutes > 60:
            hours = minutes // 60
            minutes = minutes % 60
            return (
                f"{int(hours)} hours, {int(minutes)} minutes, and {seconds:.3f} seconds"
            )
        else:
            return f"{int(minutes)} minutes and {seconds:.3f} seconds"
    else:
        return f"{seconds:.3f} seconds"
