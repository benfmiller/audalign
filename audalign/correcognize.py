import audalign.fingerprint as fingerprint
from audalign.filehandler import read, find_files
import audalign
from pydub.exceptions import CouldntDecodeError
import numpy as np
import time
import os
import scipy.signal as signal
import matplotlib.pyplot as plt


def correcognize(
    target_file_path: str,
    against_file_path: str,
    start_end_target: tuple = None,
    start_end_against: tuple = None,
    filter_matches: float = None,
    match_len_filter: int = None,
    sample_rate: int = fingerprint.DEFAULT_FS,
    max_lags: float = None,
    plot: bool = False,
    **kwargs,
):
    """Called from audalign correcognize

    Args:
        target_file_path (str): [description]
        against_file_path (str): [description]
        start_end_target (tuple, optional): [description]. Defaults to None.
        start_end_against (tuple, optional): [description]. Defaults to None.
        filter_matches (float, optional): [description]. Defaults to 0.
        match_len_filter (int, optional): [description]. Defaults to 30.
        sample_rate (int, optional): [description]. Defaults to fingerprint.DEFAULT_FS.
        max_lags (int, optional): defaults to None
        plot (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    assert (
        sample_rate < 200000
    )  # I accidentally used 441000 once... not good, big crash

    if filter_matches is None:
        filter_matches = 0.5

    print(
        f"Comparing {os.path.basename(target_file_path)} against {os.path.basename(against_file_path)}... "
    )
    t = time.time()

    target_array = read(
        target_file_path, start_end=start_end_target, sample_rate=sample_rate
    )[0]
    against_array = read(
        against_file_path, start_end=start_end_against, sample_rate=sample_rate
    )[0]

    sos = signal.butter(
        10, fingerprint.threshold, "highpass", fs=sample_rate, output="sos"
    )
    # sos = signal.butter(10, 0.125, "hp", fs=sample_rate, output="sos")
    target_array = signal.sosfilt(sos, target_array)
    against_array = signal.sosfilt(sos, against_array)

    print("Calculating correlation... ", end="")
    correlation = signal.correlate(against_array, target_array)
    scaling_factor = max(correlation)
    correlation /= np.max(np.abs(correlation), axis=0)

    results_list_tuple = find_maxes(
        correlation=correlation,
        filter_matches=filter_matches,
        match_len_filter=match_len_filter,
        max_lags=max_lags,
        sample_rate=sample_rate,
        **kwargs,
    )

    if plot:
        plot_cor(
            array_a=target_array,
            array_b=against_array,
            corr_array=correlation,
            sample_rate=sample_rate,
            arr_a_title=target_file_path,
            arr_b_title=against_file_path,
            scaling_factor=scaling_factor,
            peaks=results_list_tuple,
        )

    file_match = process_results(
        results_list=results_list_tuple,
        file_name=os.path.basename(against_file_path),
        scaling_factor=scaling_factor,
        sample_rate=sample_rate,
    )

    t = time.time() - t

    result = {}
    if file_match:
        result["match_time"] = t
        result["match_info"] = file_match
        return result

    return None


def correcognize_directory(
    target_file_path: str,
    against_directory: str,
    start_end: tuple = None,
    filter_matches: float = None,
    sample_rate: int = fingerprint.DEFAULT_FS,
    match_len_filter: int = None,
    plot: bool = False,
    max_lags: float = None,
    _file_audsegs: dict = None,
    **kwargs,
):
    """Called from audalign correcognize_directory"""
    assert (
        sample_rate < 200000
    )  # I accidentally used 441000 once... not good, big crash

    t = time.time()

    if _file_audsegs is not None:
        target_array = np.frombuffer(_file_audsegs[target_file_path]._data, np.int16)
        if filter_matches is None:
            filter_matches = 0
    else:
        target_array = read(
            target_file_path, start_end=start_end, sample_rate=sample_rate
        )[0]

    if filter_matches is None:
        filter_matches = 0.5

    sos = signal.butter(
        10, fingerprint.threshold, "highpass", fs=sample_rate, output="sos"
    )
    target_array = signal.sosfilt(sos, target_array)

    if type(against_directory) == str:
        against_files = find_files(against_directory)
    else:
        against_files = zip(against_directory, ["_"] * len(against_directory))
    file_match = {}
    for file_path, _ in against_files:

        if os.path.basename(file_path) == os.path.basename(target_file_path):
            continue
        try:
            print(
                f"Comparing {os.path.basename(target_file_path)} against {os.path.basename(file_path)}... "
            )
            if _file_audsegs is not None:
                against_array = np.frombuffer(_file_audsegs[file_path]._data, np.int16)
            else:
                against_array = read(file_path, sample_rate=sample_rate)[0]
            against_array = signal.sosfilt(sos, against_array)

            print("Calculating correlation... ", end="")
            correlation = signal.correlate(against_array, target_array)
            scaling_factor = max(correlation)
            correlation /= np.max(np.abs(correlation), axis=0)

            results_list_tuple = find_maxes(
                correlation=correlation,
                filter_matches=filter_matches,
                match_len_filter=match_len_filter,
                max_lags=max_lags,
                sample_rate=sample_rate,
                **kwargs,
            )

            if plot:
                plot_cor(
                    array_a=target_array,
                    array_b=against_array,
                    corr_array=correlation,
                    sample_rate=sample_rate,
                    arr_a_title=target_file_path,
                    arr_b_title=file_path,
                    scaling_factor=scaling_factor,
                    peaks=results_list_tuple,
                )

            single_file_match = process_results(
                results_list=results_list_tuple,
                file_name=os.path.basename(file_path),
                scaling_factor=scaling_factor,
                sample_rate=sample_rate,
            )
            if single_file_match:
                file_match = {**file_match, **single_file_match}

        except CouldntDecodeError:
            print(f'File "{file_path}" could not be decoded')

    t = time.time() - t

    result = {}
    if file_match:
        result["match_time"] = t
        result["match_info"] = file_match
        return result

    return None


def find_maxes(
    correlation: list,
    filter_matches: float,
    match_len_filter: int,
    max_lags: float,
    sample_rate: int,
    **kwargs,
) -> list:
    """This is where kwargs go. returns zip of peak indices and their heights sorted by height"""
    print("Finding Local Maximums... ", end="")
    # for more info
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    if max_lags is not None:
        max_lags = max_lags * sample_rate / 2
        if len(correlation) > 2 * max_lags:
            correlation[: int(len(correlation) / 2 - max_lags)] = 0
            correlation[int(len(correlation) / 2 + max_lags) :] = 0

    peaks, properties = signal.find_peaks(correlation, height=filter_matches, **kwargs)
    peaks -= int(len(correlation) / 2)
    peaks_tuples = zip(peaks, properties["peak_heights"])
    peaks_tuples = sorted(peaks_tuples, key=lambda x: x[1], reverse=True)

    if match_len_filter is None:
        match_len_filter = 30
    if len(peaks_tuples) > match_len_filter:
        peaks_tuples = peaks_tuples[0:match_len_filter]
    return peaks_tuples


def process_results(
    results_list: list, file_name: str, scaling_factor: float, sample_rate: int
):
    """Processes peaks and stuff into our regular recognition dictionary"""

    offset_samples = []
    offset_seconds = []
    peak_heights = []
    for result_item in results_list:
        offset_samples.append(result_item[0])
        offset_seconds.append(result_item[0] / sample_rate * 2)
        peak_heights.append(result_item[1])

    match = {}
    match[file_name] = {}

    match[file_name]["offset_samples"] = offset_samples
    match[file_name][audalign.Audalign.OFFSET_SECS] = offset_seconds
    match[file_name][audalign.Audalign.CONFIDENCE] = peak_heights
    match[file_name]["sample_rate"] = sample_rate
    match[file_name]["scaling_factor"] = scaling_factor

    print("done")

    if len(match[file_name]["offset_seconds"]) > 0:
        return match
    return None


# ---------------------------------------------------------------------------------


def plot_cor(
    array_a,
    array_b,
    corr_array,
    sample_rate,
    title="Comparison",
    arr_a_title=None,
    arr_b_title=None,
    peaks=None,
    scaling_factor=None,
):
    """
    Really nifty plotter, lots of good information here.
    Can get really slow if the sample rate is high and the audio file is long.
    """
    new_vis_wsize = int(fingerprint.DEFAULT_WINDOW_SIZE / 44100 * sample_rate)
    fig = plt.figure(title)

    fig.add_subplot(3, 2, 1)
    plt.plot(array_a)
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    if arr_a_title:
        plt.title(arr_a_title)

    arr2d_a = fingerprint.fingerprint(
        array_a, fs=sample_rate, wsize=new_vis_wsize, retspec=True
    )
    fig.add_subplot(3, 2, 2)
    plt.imshow(arr2d_a)  # , cmap=plt.cm.gray)
    plt.gca().invert_yaxis()
    if arr_a_title:
        plt.title(arr_a_title)

    fig.add_subplot(3, 2, 3)
    plt.plot(array_b)
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    if arr_b_title:
        plt.title(arr_b_title)

    arr2d_b = fingerprint.fingerprint(
        array_b, fs=sample_rate, wsize=new_vis_wsize, retspec=True
    )
    fig.add_subplot(3, 2, 4)
    plt.imshow(arr2d_b)
    plt.gca().invert_yaxis()
    if arr_b_title:
        plt.title(arr_b_title)

    fig.add_subplot(3, 2, 5)
    plt.plot(corr_array)
    if scaling_factor:
        plt.title(f"Correlation - Scaling Factor: {scaling_factor}")
    else:
        plt.title(f"Correlation")
    plt.xlabel("Sample Index")
    plt.ylabel("correlation")
    if peaks:
        indexes = [x[0] + int(len(corr_array) / 2) for x in peaks]
        heights = [x[1] for x in peaks]
        plt.plot(indexes, heights, "x")

    fig.tight_layout()

    plt.show()