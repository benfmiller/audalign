from os.path import normcase
import audalign.fingerprint as fingerprint
from audalign.filehandler import read
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
    filter_matches: float = 0,
    match_len_filter: int = 30,
    sample_rate: int = fingerprint.DEFAULT_FS,
    plot: bool = False,
    **kwargs,
):
    assert (
        sample_rate < 200000
    )  # I accidentally used 441000 once... not good, big crash

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

    correlation = signal.correlate(target_array, against_array)
    scaling_factor = max(correlation)
    correlation /= np.max(np.abs(correlation), axis=0)

    if plot:
        plot_cor(  # add results list option for scatter of maxes???
            array_a=target_array,
            array_b=against_array,
            corr_array=correlation,
            sample_rate=sample_rate,
            arr_a_title=target_file_path,
            arr_b_title=against_file_path,
        )

    results_list_tuple = find_maxes(
        correlation=correlation,
        filter_matches=filter_matches,
        match_len_filter=match_len_filter,
        **kwargs,
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
    filter_matches: float = 0,
    sample_rate: int = fingerprint.DEFAULT_FS,
    plot: bool = False,
    **kwargs,
):
    print(f"{target_file_path} : {against_directory}")
    # TODO correcognize directory
    ...


def find_maxes(
    correlation: list, filter_matches: float, match_len_filter: int, **kwargs
) -> list:
    # for more info
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    peaks, properties = signal.find_peaks(correlation, height=filter_matches, **kwargs)
    peaks -= int(len(correlation) / 2)
    peaks_tuples = zip(peaks, properties["peak_heights"])
    peaks_tuples = sorted(peaks_tuples, key=lambda x: x[1], reverse=True)
    if len(peaks_tuples) > match_len_filter:
        peaks_tuples = peaks_tuples[0:match_len_filter]
    return peaks_tuples


def process_results(
    results_list: list, file_name: str, scaling_factor: float, sample_rate: int
):

    return None
    # TODO
    print("Calculating results... ", end="")
    i = 0  # remove bad results or results below threshold
    while i < len(results_list):
        if results_list[i][2][0] == 10000000 or results_list[i][2][1] == 10000000:
            results_list.pop(i)
            continue
        i += 1

    offset_dict = {}  # aggregate results by time difference
    for i in results_list:
        if i[0] - i[1] not in offset_dict:
            offset_dict[i[0] - i[1]] = [0, 0, 0]  # mse,ssim,total
        temp_result = offset_dict[i[0] - i[1]]
        temp_result[0] += i[2][0]
        temp_result[1] += i[2][1]
        temp_result[2] += 1
        offset_dict[i[0] - i[1]] = temp_result

    for i in offset_dict.keys():  # average mse and ssim
        temp_result = offset_dict[i]
        temp_result[0] /= temp_result[2]
        temp_result[1] /= temp_result[2]
        offset_dict[i] = temp_result

    match_offsets = []
    for t_difference, match_data in offset_dict.items():
        match_offsets.append((match_data, t_difference))
    match_offsets = sorted(
        match_offsets,
        reverse=True,
        key=lambda x: (np.log2(x[0][2] + 1) * (np.log(x[0][1] + 1) / np.log(1.5))),
    )  # sort by ssim must be reversed for ssim
    # match_offsets, reverse=True, key=lambda x: (x[0][2], x[0][1])
    # match_offsets, reverse=True, key=lambda x: x[0][2] sorts by num matches
    # match_offsets, reverse=True, key=lambda x: (x[0][2], x[0][1]) sorts once by ssim, then finally by num matches
    # math.log

    offset_count = []
    offset_diff = []
    offset_ssim = []
    offset_mse = []
    for i in match_offsets:
        offset_count.append(i[0][2])
        offset_diff.append(i[1])
        offset_ssim.append(i[0][1])
        offset_mse.append(i[0][0])

    match = {}
    match[filename] = {}

    match[filename]["num_matches"] = offset_count
    match[filename]["offset_samples"] = offset_diff
    match[filename]["ssim"] = offset_ssim
    match[filename]["mse"] = offset_mse

    offset_seconds = []
    for i in offset_diff:
        nseconds = round(
            float(i)
            / fingerprint.DEFAULT_FS
            * fingerprint.DEFAULT_WINDOW_SIZE
            * fingerprint.DEFAULT_OVERLAP_RATIO,
            5,
        )
        offset_seconds.append(nseconds)

    match[filename]["offset_seconds"] = offset_seconds
    print("done")

    if len(match[filename]["offset_seconds"]) > 0:
        return match
    return None

    # xin, fs = sf.read('recording1.wav')
    # frame_len = int(fs*5*1e-3)
    # dim_x =xin.shape
    # M = dim_x[0] # No. of rows
    # N= dim_x[1] # No. of col
    # sample_lim = frame_len*100
    # tau = [0]
    # M_lim = 20000 # for testing as processing takes time
    # for i in range(1,N):
    #     c = np.correlate(xin[0:M_lim,0],xin[0:M_lim,i],"full")
    #     maxlags = M_lim-1
    #     c = c[M_lim -1 -maxlags: M_lim + maxlags]
    #     Rmax_pos = np.argmax(c)
    #     pos = Rmax_pos-M_lim+1
    #     tau.append(pos)
    # print(tau)


# ---------------------------------------------------------------------------------


def plot_cor(
    array_a,
    array_b,
    corr_array,
    sample_rate,
    title="Comparison",
    arr_a_title=None,
    arr_b_title=None,
):
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
    plt.title(f"Correlation")
    plt.xlabel("Sample Index")
    plt.ylabel("correlation")

    fig.tight_layout()

    plt.show()