import audalign.fingerprint as fingerprint
from audalign.filehandler import read, find_files, get_shifted_file
import audalign
from pydub.exceptions import CouldntDecodeError
import numpy as np
import tqdm
import time
import os
import scipy.signal as signal
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial

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
    technique: str = "correlation",
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
    if technique not in ["correlation", "correlation_spectrogram"]:
        raise ValueError(
            "technique must be either 'correlation' or 'correlation_spectrogram"
        )

    if max_lags is not None:
        max_lags = int(calc_spec_windows(max_lags, sample_rate, technique))
    if filter_matches is None:
        filter_matches = 0.0
    if locality is not None:
        locality = int(calc_spec_windows(locality, sample_rate, technique))
    if locality_filter_prop is None:
        locality_filter_prop = DEFAULT_LOCALITY_FILTER_PROP
    elif locality_filter_prop > 1.0:
        locality_filter_prop = 1.0
    elif locality_filter_prop < 0:
        locality_filter_prop = 0.0

    sos = signal.butter(
        10, fingerprint.threshold, "highpass", fs=sample_rate, output="sos"
    )
    # sos = signal.butter(10, 0.125, "hp", fs=sample_rate, output="sos")
    target_array = get_array(
        target_file_path,
        start_end=start_end_target,
        sample_rate=sample_rate,
        _file_audsegs=None,
        technique=technique,
        sos=sos,
    )
    against_array = get_array(
        against_file_path,
        start_end=start_end_against,
        sample_rate=sample_rate,
        _file_audsegs=None,
        technique=technique,
        sos=sos,
    )

    t = time.time()
    file_match = _correcognize(
        target_array=target_array,
        target_file_path=target_file_path,
        against_array=against_array,
        against_file_path=against_file_path,
        filter_matches=filter_matches,
        match_len_filter=match_len_filter,
        locality=locality,
        locality_filter_prop=locality_filter_prop,
        sample_rate=sample_rate,
        max_lags=max_lags,
        plot=plot,
        technique=technique,
        **kwargs,
    )
    t = time.time() - t

    result = {}
    if len(file_match) > 0:
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
    _include_filename: bool = False,
    technique: str = "correlation",
    use_multiprocessing: bool = True,
    num_processes: int = None,
    **kwargs,
):
    """Called from audalign correcognize_directory"""
    assert (
        sample_rate < 200000
    )  # I accidentally used 441000 once... not good, big crash
    if technique not in ["correlation", "correlation_spectrogram"]:
        raise ValueError(
            "technique must be either 'correlation' or 'correlation_spectrogram"
        )

    t = time.time()

    if max_lags is not None:
        max_lags = int(calc_spec_windows(max_lags, sample_rate, technique))
    if locality is not None:
        locality = int(calc_spec_windows(locality, sample_rate, technique))
    if locality_filter_prop is None:
        locality_filter_prop = DEFAULT_LOCALITY_FILTER_PROP
    elif locality_filter_prop > 1.0:
        locality_filter_prop = 1.0
    elif locality_filter_prop < 0:
        locality_filter_prop = 0.0
    if _file_audsegs is not None and filter_matches is None:
        filter_matches = 0.0
    elif filter_matches is None:
        filter_matches = 0.0

    if fingerprint.threshold <= 0:
        fingerprint.threshold = 1
    sos = signal.butter(
        10, fingerprint.threshold, "highpass", fs=sample_rate, output="sos"
    )

    if use_multiprocessing is False:
        target_array = get_array(
            target_file_path,
            start_end,
            sample_rate=sample_rate,
            _file_audsegs=_file_audsegs,
            technique=technique,
            sos=sos,
        )
        target_file_path = (target_file_path, target_array)

    if type(against_directory) == str:
        against_files = find_files(against_directory)
    else:
        against_files = zip(against_directory, ["_"] * len(against_directory))

    _correcognize_dir_ = partial(
        _correcognize_dir,
        target_file_path=target_file_path,
        start_end=start_end,
        _file_audsegs=_file_audsegs,
        sample_rate=sample_rate,
        technique=technique,
        sos_filter=sos,
        filter_matches=filter_matches,
        match_len_filter=match_len_filter,
        locality=locality,
        locality_filter_prop=locality_filter_prop,
        max_lags=max_lags,
        plot=plot,
        **kwargs,
    )

    if use_multiprocessing == False:
        results_list = []
        for file_path in against_files:
            results_list += [_correcognize_dir_(file_path)]
    else:
        try:
            nprocesses = num_processes or multiprocessing.cpu_count()
        except NotImplementedError:
            nprocesses = 1
        else:
            nprocesses = 1 if nprocesses <= 0 else nprocesses

        with multiprocessing.Pool(nprocesses) as pool:
            results_list = pool.map(_correcognize_dir_, tqdm.tqdm(list(against_files)))
            pool.close()
            pool.join()

    file_match = {}
    for i in results_list:
        file_match = {**file_match, **i}

    t = time.time() - t

    result = {}
    if len(file_match) > 0:
        result["match_time"] = t
        result["match_info"] = file_match
        if _include_filename:
            if type(target_file_path) == tuple:
                result["filename"] = target_file_path[0]
            else:
                result["filename"] = target_file_path
        return result
    return None


def _correcognize(
    target_array: list,
    target_file_path: str,
    against_array: list,
    against_file_path: str,
    filter_matches: float = None,
    match_len_filter: int = None,
    locality: float = None,
    locality_filter_prop: float = None,
    sample_rate: int = fingerprint.DEFAULT_FS,
    max_lags: float = None,
    plot: bool = False,
    technique: str = "correlation",
    **kwargs,
):
    print(
        f"Comparing {os.path.basename(target_file_path)} against {os.path.basename(against_file_path)}... "
    )

    print("Calculating correlation... ", end="")
    indexes = (
        find_index_arr(against_array, target_array, locality, max_lags)
        if locality is not None
        else []
    )
    correlation = calc_corrs(
        against_array,
        target_array,
        locality=locality,
        technique=technique,
        indexes=indexes,
    )

    if locality is None:
        correlation = list(correlation)[0]
    results_list_tuple, scaling_factor = find_maxes(
        correlation=correlation,
        filter_matches=filter_matches,
        match_len_filter=match_len_filter,
        max_lags=max_lags,
        locality_filter_prop=locality_filter_prop,
        locality=locality,
        technique=technique,
        indexes_len=len(indexes),
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
            technique=technique,
        )
    elif plot:
        print("\nCorrelation Plot not compatible with locality")
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

    return process_results(
        results_list=results_list_tuple,
        file_name=os.path.basename(against_file_path),
        scaling_factor=scaling_factor,
        sample_rate=sample_rate,
        locality=locality,
        technique=technique,
    )


def _correcognize_dir(
    against_file_path,
    target_file_path,
    start_end,
    _file_audsegs,
    sample_rate,
    technique,
    sos_filter,
    filter_matches,
    match_len_filter,
    locality,
    locality_filter_prop,
    max_lags,
    plot,
    **kwargs,
):
    if type(target_file_path) == str:
        target_array = get_array(
            target_file_path,
            start_end,
            sample_rate=sample_rate,
            _file_audsegs=_file_audsegs,
            technique=technique,
            sos=sos_filter,
        )
    else:
        target_file_path, target_array = target_file_path

    against_file_path, _ = against_file_path

    if os.path.basename(against_file_path) == os.path.basename(target_file_path):
        return {}
    try:
        against_array = get_array(
            against_file_path,
            start_end=None,
            sample_rate=sample_rate,
            _file_audsegs=_file_audsegs,
            technique=technique,
            sos=sos_filter,
        )
        return _correcognize(
            target_array=target_array,
            target_file_path=target_file_path,
            against_array=against_array,
            against_file_path=against_file_path,
            filter_matches=filter_matches,
            match_len_filter=match_len_filter,
            locality=locality,
            locality_filter_prop=locality_filter_prop,
            sample_rate=sample_rate,
            max_lags=max_lags,
            plot=plot,
            technique=technique,
            **kwargs,
        )

    except CouldntDecodeError:
        print(f'File "{against_file_path}" could not be decoded')
        return {}


def get_array(
    file_path,
    start_end,
    sample_rate,
    _file_audsegs,
    technique,
    sos,
):
    if _file_audsegs is not None:
        target_array = get_shifted_file(
            file_path,
            _file_audsegs[file_path],
            sample_rate=sample_rate,
        )
    else:
        target_array = read(file_path, start_end=start_end, sample_rate=sample_rate)[0]
    if sos is not None:
        target_array = signal.sosfilt(sos, target_array)
    if technique == "correlation_spectrogram":
        target_array = fingerprint.fingerprint(
            target_array, fs=sample_rate, retspec=True
        ).T
        target_array = np.clip(target_array, 0, 500)  # never louder than 500
    return target_array


def calc_array_indexes(array, locality):
    index_list = []
    if locality > len(array):
        index_list += [0]
    else:
        [
            index_list.append(i)
            for i in range(
                0, len(array) - int(locality), int(locality * (1 - OVERLAP_RATIO))
            )
        ]
        if (
            len(array) - int(locality) not in index_list
            and len(array) - int(locality) > 0
        ):
            index_list.append(len(array) - int(locality))
    return index_list


def frames_to_sec(frames, sample_rate):
    return round(
        float(frames)
        / sample_rate
        * fingerprint.DEFAULT_WINDOW_SIZE
        * fingerprint.DEFAULT_OVERLAP_RATIO,
        5,
    )


def calc_spec_windows(seconds, sample_rate, technique):
    if technique == "correlation":
        return seconds * sample_rate
    elif technique == "correlation_spectrogram":
        return max(
            int(
                seconds
                // (
                    fingerprint.DEFAULT_WINDOW_SIZE
                    / sample_rate
                    * fingerprint.DEFAULT_OVERLAP_RATIO
                )
            ),
            1,
        )


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
                if abs(j - i) <= max_lags + locality - 1:
                    index_pairs += [(i, j)]
    return index_pairs


def calc_corrs(
    against_array,
    target_array,
    locality: float,
    technique: str,
    indexes: list,
):
    if locality is None:
        if technique == "correlation":
            yield (
                signal.correlate(against_array, target_array),
                (against_array.size, target_array.size),
            )
        elif technique == "correlation_spectrogram":
            against_array = against_array.T
            target_array = target_array.T
            yield (
                _calc_corrs_spec(against_array, target_array),
                (against_array.shape[1], target_array.shape[1]),
            )
    else:
        locality_a = len(against_array) if locality > len(against_array) else locality
        locality_b = len(target_array) if locality > len(target_array) else locality
        if technique == "correlation":
            for pair in indexes:
                yield [
                    (
                        signal.correlate(
                            against_array[pair[0] : pair[0] + locality_a],
                            target_array[pair[1] : pair[1] + locality_b],
                        ),
                        (locality_a, locality_b),
                    ),
                    pair,
                ]
        elif technique == "correlation_spectrogram":
            for pair in indexes:
                yield [
                    (
                        _calc_corrs_spec(
                            against_array[pair[0] : pair[0] + locality_a].T,
                            target_array[pair[1] : pair[1] + locality_b].T,
                        ),
                        (locality_a, locality_b),
                    ),
                    pair,
                ]


def _calc_corrs_spec(against, target):
    corr_result = signal.correlate(against[0], target[0])
    for i in range(1, len(against)):
        corr_result += signal.correlate(against[i], target[i])
    return corr_result


def find_maxes(
    correlation: list,
    filter_matches: float,
    match_len_filter: int,
    max_lags: float,
    locality_filter_prop: float,
    locality: float,
    technique: str,
    indexes_len: int,
    **kwargs,
) -> list:
    print("Finding Local Maximums... ", end="")
    if locality is None:
        return _find_peaks(
            correlation=correlation[0],
            len_tups=correlation[1],
            filter_matches=filter_matches,
            match_len_filter=match_len_filter,
            max_lags=max_lags,
            technique=technique,
            **kwargs,
        )
    else:
        total_peaks, peak_indexes = [], []
        for i in tqdm.tqdm(correlation, total=indexes_len):
            total_peaks += [
                _find_peaks(
                    correlation=i[0][0],
                    len_tups=i[0][1],
                    filter_matches=0,
                    match_len_filter=match_len_filter,
                    max_lags=max_lags,
                    index_pair=i[1],
                    technique=technique,
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
    len_tups: tuple,
    filter_matches: float,
    match_len_filter: int,
    max_lags: float,
    index_pair: tuple = None,
    technique: str = "correlation",
    **kwargs,
):
    """This is where kwargs go. returns zip of peak indices and their heights sorted by height"""
    max_corr = np.max(correlation)
    if technique == "correlation":
        scaling_factor = max_corr / len(correlation) / SCALING_16_BIT
    elif technique == "correlation_spectrogram":
        scaling_factor = (
            max_corr / len(correlation) / (fingerprint.DEFAULT_WINDOW_SIZE / 2) / 50
        )  # 200 is about the max of spectrogram
        # *4 because most frequency bands are quite low, especially with butter filter
    correlation = (
        correlation / np.max(np.abs(correlation), axis=0)
        if max_corr > 0
        else correlation
    )
    lag_array = signal.correlation_lags(len_tups[0], len_tups[1], mode="full")
    if max_lags is not None:
        shift = 0
        if index_pair is not None:
            shift = index_pair[0] - index_pair[1]
        if np.max(lag_array) > max_lags - shift:
            correlation[np.where(lag_array == (max_lags - shift))[0][0] + 1 :] = 0
        if np.min(lag_array) < -max_lags - shift:
            correlation[: np.where(lag_array == (-max_lags - shift))[0][0]] = 0

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    peaks, properties = signal.find_peaks(correlation, height=filter_matches, **kwargs)
    peaks = [lag_array[x] for x in peaks]
    peaks_tuples = zip(peaks, properties["peak_heights"])
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
    locality: float = None,
    technique: str = "correlation",
):
    """Processes peaks and stuff into our regular recognition dictionary"""

    offset_samples = []
    offset_seconds = []
    peak_heights = []
    locality_list = []
    locality_seconds = []
    if locality is None:
        for result_item in results_list:
            offset_samples.append(result_item[0])
            peak_heights.append(result_item[1])
            if technique == "correlation":
                offset_seconds.append(result_item[0] / sample_rate)
            elif technique == "correlation_spectrogram":
                offset_seconds.append(frames_to_sec(result_item[0], sample_rate))
        locality_list = [None] * len(results_list)
        locality_seconds = [None] * len(results_list)
    else:
        for result_item in results_list:
            offset_samples.append(result_item[0][0])
            if technique == "correlation":
                offset_seconds.append(result_item[0][0] / sample_rate)
            elif technique == "correlation_spectrogram":
                offset_seconds.append(frames_to_sec(result_item[0][0], sample_rate))
            peak_heights.append(result_item[0][1])
            locality_list.append(result_item[1])
            if technique == "correlation":
                locality_seconds.append(
                    [
                        (
                            x[1] / sample_rate,
                            x[0] / sample_rate,
                            x[2],
                        )  # target, against, scaling
                        for x in result_item[1]
                    ]
                )
            elif technique == "correlation_spectrogram":
                locality_seconds.append(
                    [
                        (
                            frames_to_sec(x[1], sample_rate),
                            frames_to_sec(x[0], sample_rate),
                            x[2],
                        )  # target, against, scaling
                        for x in result_item[1]
                    ]
                )

    match = {}
    match[file_name] = {}

    if technique == "correlation":
        match[file_name]["locality_samples"] = locality_list
        match[file_name]["offset_samples"] = offset_samples
    elif technique == "correlation_spectrogram":
        match[file_name][audalign.Audalign.LOCALITY] = locality_list
        match[file_name][audalign.Audalign.OFFSET_SAMPLES] = offset_samples
    match[file_name][audalign.Audalign.LOCALITY_SECS] = locality_seconds
    match[file_name][audalign.Audalign.OFFSET_SECS] = offset_seconds
    match[file_name][audalign.Audalign.CONFIDENCE] = peak_heights
    match[file_name]["sample_rate"] = sample_rate
    match[file_name]["scaling_factor"] = scaling_factor

    print("done")

    if len(match[file_name]["offset_seconds"]) > 0:
        return match
    return {}


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
    technique=None,
):
    """
    Really nifty plotter, lots of good information here.
    Can get really slow if the sample rate is high and the audio file is long.
    """
    new_vis_wsize = int(fingerprint.DEFAULT_WINDOW_SIZE / 44100 * sample_rate)
    fig = plt.figure(title)

    if technique == "correlation":
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
    elif technique == "correlation_spectrogram":
        fig.add_subplot(3, 2, 1)
        plt.imshow(array_a)  # , cmap=plt.cm.gray)
        plt.gca().invert_yaxis()
        if arr_a_title:
            plt.title(arr_a_title)

        fig.add_subplot(3, 2, 3)
        plt.imshow(array_b)
        plt.gca().invert_yaxis()
        if arr_b_title:
            plt.title(arr_b_title)

    if corr_array is not None:
        max_cor = np.max(corr_array[0])

        fig.add_subplot(3, 2, 5)
        plt.plot(corr_array[0])
        if scaling_factor:
            plt.title(f"Correlation - Scaling Factor: {scaling_factor}")
        else:
            plt.title(f"Correlation")
        plt.xlabel("Sample Index")
        plt.ylabel("correlation")
        if peaks:
            indexes = [x[0] / 2 + int(len(corr_array[0]) / 2) for x in peaks]
            heights = [x[1] * max_cor for x in peaks]
            plt.plot(indexes, heights, "x")

    fig.tight_layout()

    plt.show()