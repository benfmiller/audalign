from numpy.lib.function_base import append
import audalign.fingerprint as fingerprint
from audalign.filehandler import read, find_files, get_shifted_file
import audalign
from pydub.exceptions import CouldntDecodeError
import numpy as np
import time
import os
import scipy.signal as signal
import matplotlib.pyplot as plt

# For locality window overlaps
SCALING_16_BIT = 65536
OVERLAP_RATIO = 0.5
DEFAULT_LOCALITY_FILTER_PROP = 0.6


def correcognize(
    target_file_path: str,
    against_file_path: str,
    start_end_target: tuple = None,
    start_end_against: tuple = None,
    filter_matches: float = None,
    match_len_filter: int = None,
    locality: float = None,
    locality_filter_prop: float = None,
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

    if max_lags is not None:
        max_lags = max_lags * sample_rate / 2
    if filter_matches is None:
        filter_matches = 0.5
    if locality is not None:
        locality = locality * sample_rate
    if locality_filter_prop is None:
        locality_filter_prop = DEFAULT_LOCALITY_FILTER_PROP
    elif locality_filter_prop > 1.0:
        locality_filter_prop = 1.0
    elif locality_filter_prop < 0:
        locality_filter_prop = 0.0

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
    correlation = calc_corrs(
        against_array, target_array, locality=locality, max_lags=max_lags
    )

    results_list_tuple, scaling_factor = find_maxes(
        correlation=correlation,
        filter_matches=filter_matches,
        match_len_filter=match_len_filter,
        max_lags=max_lags,
        locality_filter_prop=locality_filter_prop,
        locality=locality,
        **kwargs,
    )

    if plot and locality is None:
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
    elif plot:
        print("Correlation Plot not compatible with locality")
        plot_cor(
            array_a=target_array,
            array_b=against_array,
            corr_array=None,
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
        locality=locality,
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
    locality: float = None,
    locality_filter_prop: float = None,
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

    if max_lags is not None:
        max_lags = max_lags * sample_rate / 2
    if locality is not None:
        locality = locality * sample_rate
    if locality_filter_prop is None:
        locality_filter_prop = DEFAULT_LOCALITY_FILTER_PROP
    elif locality_filter_prop > 1.0:
        locality_filter_prop = 1.0
    elif locality_filter_prop < 0:
        locality_filter_prop = 0.0

    if _file_audsegs is not None:
        target_array = _file_audsegs[target_file_path]
        # target_array = get_shifted_file( # might want for multiprocessing in the future
        #     target_file_path, _file_audsegs[target_file_path], sample_rate=sample_rate
        # )
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
                against_array = _file_audsegs[file_path]
                # against_array = get_shifted_file( # might want for multiprocessing in the future
                #     file_path, _file_audsegs[file_path], sample_rate=sample_rate
                # )
            else:
                against_array = read(file_path, sample_rate=sample_rate)[0]
            against_array = signal.sosfilt(sos, against_array)

            print("Calculating correlation... ", end="")
            correlation = calc_corrs(
                against_array, target_array, locality=locality, max_lags=max_lags
            )

            results_list_tuple, scaling_factor = find_maxes(
                correlation=correlation,
                filter_matches=filter_matches,
                match_len_filter=match_len_filter,
                max_lags=max_lags,
                locality_filter_prop=locality_filter_prop,
                locality=locality,
                **kwargs,
            )

            if plot and locality is None:
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
            elif plot:
                print("Correlation Plot not compatible with locality")
                plot_cor(
                    array_a=target_array,
                    array_b=against_array,
                    corr_array=None,
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
                locality=locality,
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


def calc_array_indexes(array, locality):
    index_list = []
    if locality > len(array):
        index_list += [0]
    else:
        [
            index_list.append(i)
            for i in range(0, len(array) - int(locality), int(locality * OVERLAP_RATIO))
        ]
        if len(array) - int(locality) not in index_list:
            index_list.append(len(array) - int(locality))
    return index_list


def find_index_arr(against_array, target_array, locality, max_lags):
    index_list_against = calc_array_indexes(against_array, locality)
    index_list_target = calc_array_indexes(target_array, locality)
    index_pairs = []
    if max_lags is None:
        for i in index_list_against:
            for j in index_list_target:
                index_pairs += [(i, j)]
    else:
        for i in index_list_against:
            for j in index_list_target:
                if abs(j - i) <= max_lags * 2 + locality:
                    index_pairs += [(i, j)]
    return index_pairs


def calc_corrs(against_array, target_array, locality: float, max_lags: float):
    if locality is None:
        yield signal.correlate(against_array, target_array)
    else:
        locality_a = len(against_array) if locality > len(against_array) else locality
        locality_b = len(target_array) if locality > len(target_array) else locality
        indexes = find_index_arr(against_array, target_array, locality, max_lags)
        for pair in indexes:
            yield [
                signal.correlate(
                    against_array[pair[0] : pair[0] + locality_a],
                    target_array[pair[1] : pair[1] + locality_b],
                ),
                pair,
            ]


def find_maxes(
    correlation: list,
    filter_matches: float,
    match_len_filter: int,
    max_lags: float,
    locality_filter_prop: float,
    locality: float,
    **kwargs,
) -> list:
    print("Finding Local Maximums... ", end="")
    if locality is None:
        correlation = list(correlation)[0]
        return _find_peaks(
            correlation=correlation,
            filter_matches=filter_matches,
            match_len_filter=match_len_filter,
            max_lags=max_lags,
            **kwargs,
        )
    else:
        total_peaks, peak_indexes = [], []
        for i in correlation:
            total_peaks += [
                _find_peaks(
                    correlation=i[0],
                    filter_matches=0,
                    # filter_matches=filter_matches,
                    match_len_filter=match_len_filter,
                    max_lags=max_lags,
                    index_pair=i[1],
                    **kwargs,
                ),
            ]
            peak_indexes += [i[1]]
        return process_loc_peaks(
            total_peaks=total_peaks,
            peak_indexes=peak_indexes,
            locality_filter_prop=locality_filter_prop,
            match_len_filter=match_len_filter,
            filter_matches=filter_matches,
        )


def _find_peaks(
    correlation: list,
    filter_matches: float,
    match_len_filter: int,
    max_lags: float,
    index_pair: tuple = None,
    **kwargs,
):
    """This is where kwargs go. returns zip of peak indices and their heights sorted by height"""
    scaling_factor = max(correlation) / len(correlation) / SCALING_16_BIT
    correlation = (
        correlation / np.max(np.abs(correlation), axis=0)
        if max(correlation) > 0
        else correlation
    )
    # This is quite a bit faster, but I couldn't get it to work. Just wasn't worth it.
    # Fix this for speedup with locality and max_lags

    # if max_lags is not None and index_pair is not None:
    #     shift = index_pair[0] - index_pair[1]
    #     # lag_indexes = [0, 0]
    #     if len(correlation) + shift > 2 * max_lags:
    #         correlation[int(len(correlation) / 2 + max_lags - (shift / 2)) + 1 :] = 0
    #     if len(correlation) - shift > 2 * max_lags:
    #         correlation[: int(len(correlation) / 2 - max_lags + (shift / 2))] = 0
    #     # correlation = correlation[lag_indexes[0] : lag_indexes[1]]
    if max_lags is not None and index_pair is None:
        if len(correlation) > 2 * max_lags:
            correlation[: int(len(correlation) / 2 - max_lags)] = 0
            correlation[int(len(correlation) / 2 + max_lags) + 1 :] = 0

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    peaks, properties = signal.find_peaks(correlation, height=filter_matches, **kwargs)
    peaks -= int(len(correlation) / 2)
    peaks *= 2
    peaks_tuples = zip(peaks, properties["peak_heights"])
    if max_lags is not None and index_pair is not None:
        peaks_tuples = sorted(peaks_tuples, key=lambda x: x[0])
        lag_indexes = [0, len(peaks_tuples)]
        shift = index_pair[0] - index_pair[1]
        for i, peak in enumerate(peaks_tuples):
            if peak[0] + shift < -max_lags * 2:
                lag_indexes[0] = i
            if peak[0] + shift > max_lags * 2:
                lag_indexes[1] = i
                break
        peaks_tuples = peaks_tuples[lag_indexes[0] : lag_indexes[1]]

    peaks_tuples = sorted(peaks_tuples, key=lambda x: x[1], reverse=True)

    if match_len_filter is None:
        match_len_filter = 30
    if len(peaks_tuples) > match_len_filter:
        peaks_tuples = peaks_tuples[:match_len_filter]
    return peaks_tuples, scaling_factor


def process_loc_peaks(
    total_peaks, peak_indexes, locality_filter_prop, match_len_filter, filter_matches
):
    print("Processing Peaks... ", end="")
    if match_len_filter is None:
        match_len_filter = 30

    shift_dict = {}
    max_scaling_factor = 0
    for i, peaks in enumerate(
        total_peaks
    ):  # combining locality matches into list of peaks with info
        shift = peak_indexes[i][0] - peak_indexes[i][1]
        scaling_factor = peaks[1]  # sf of match
        peaks = peaks[0]
        for peak in peaks:
            if peak[0] + shift not in shift_dict:
                shift_dict[peak[0] + shift] = []
            shift_dict[peak[0] + shift].append(
                [
                    peak_indexes[i][0],
                    peak_indexes[i][1],
                    peak[1] * scaling_factor,
                ]
            )
        if scaling_factor > max_scaling_factor:
            max_scaling_factor = scaling_factor
    peaks_tuples_tuples = []
    for offset in shift_dict.keys():
        temp_list = shift_dict[offset]
        top_scaling_factor = max(temp_list, key=lambda x: x[2])[2]
        if top_scaling_factor / max_scaling_factor < filter_matches:
            continue
        i = 0
        while i < len(temp_list):
            if temp_list[i][2] < top_scaling_factor * locality_filter_prop:
                temp_list.pop(i)
                continue
            i += 1
        temp_list = [(x[0], x[1], x[2] / max_scaling_factor) for x in temp_list]
        temp_list = sorted(temp_list, key=lambda x: x[2], reverse=True)
        if len(temp_list) > match_len_filter:
            temp_list = temp_list[:match_len_filter]
        peaks_tuples_tuples.append(
            [[offset, top_scaling_factor / max_scaling_factor], temp_list]
        )
    peaks_tuples_tuples = sorted(
        peaks_tuples_tuples, key=lambda x: x[0][1], reverse=True
    )
    if len(peaks_tuples_tuples) > match_len_filter:
        peaks_tuples_tuples = peaks_tuples_tuples[:match_len_filter]

    return peaks_tuples_tuples, max_scaling_factor


def process_results(
    results_list: list,
    file_name: str,
    scaling_factor: float,
    sample_rate: int,
    locality=None,
):
    """Processes peaks and stuff into our regular recognition dictionary"""

    offset_samples = []
    offset_seconds = []
    peak_heights = []
    locality_list = []
    locality_seconds = []
    if locality is None:
        locality_list = [None] * len(results_list)
        locality_seconds = [None] * len(results_list)
        for result_item in results_list:
            offset_samples.append(result_item[0])
            offset_seconds.append(result_item[0] / sample_rate)
            peak_heights.append(result_item[1])
    else:
        for result_item in results_list:
            offset_samples.append(result_item[0][0])
            offset_seconds.append(result_item[0][0] / sample_rate)
            peak_heights.append(result_item[0][1])
            locality_list.append(result_item[1])
            locality_seconds.append(
                [
                    (
                        x[1] / sample_rate * 2,
                        x[0] / sample_rate * 2,
                        x[2],
                    )  # against target or target against???
                    for x in result_item[1]
                ]
            )

    match = {}
    match[file_name] = {}

    match[file_name]["locality_samples"] = locality_list
    match[file_name][audalign.Audalign.LOCALITY_SECS] = locality_seconds
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

    if corr_array is not None:
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