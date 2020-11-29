import time
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from functools import partial

import audalign.fingerprint as fingerprint
from audalign.filehandler import read


def get_frame_width_and_overlap(seconds_width: float, overlap_ratio: float):
    seconds_width = max(
        int(
            seconds_width
            // (
                fingerprint.DEFAULT_WINDOW_SIZE
                / fingerprint.DEFAULT_FS
                * fingerprint.DEFAULT_OVERLAP_RATIO
            )
        ),
        1,
    )
    overlap_ratio = max(int(overlap_ratio * seconds_width), 1)
    return seconds_width, overlap_ratio


def calculate_comp_values(
    index_tuple, img_width=0, target_arr2d=[[]], against_arr2d=[[]], threshold=215
):
    # print(np.amax(target_arr2d[index_tuple[0] : index_tuple[0] + img_width]))
    # array.mean() very small range of values, usually between 0.4 and 2
    # Plus, finding the max only uses regions with large peaks, which could reduce
    # noisy secions being included.
    try:
        if (
            np.amax(target_arr2d[index_tuple[0] : index_tuple[0] + img_width])
            < threshold
            or np.amax(against_arr2d[index_tuple[1] : index_tuple[1] + img_width])
            < threshold
        ):
            return (index_tuple[0], index_tuple[1], (10000000, 10000000))
        m = mean_squared_error(
            target_arr2d[index_tuple[0] : index_tuple[0] + img_width],
            against_arr2d[index_tuple[1] : index_tuple[1] + img_width],
        )
        s = ssim(
            target_arr2d[index_tuple[0] : index_tuple[0] + img_width],
            against_arr2d[index_tuple[1] : index_tuple[1] + img_width],
        )
        return (index_tuple[0], index_tuple[1], (m, s))
    except ZeroDivisionError as e:
        m = 10000000
        print(f"zero division error for index {index_tuple} and img width{img_width}")
        s = 10000000
        return (index_tuple[0], index_tuple[1], (m, s))


def visrecognize(
    target_file_path: str,
    against_file_path: str,
    img_width=1.0,
    overlap_ratio=0.5,
    volume_threshold=215,
    use_multiprocessing=True,
    num_processes=None,
    plot=False,
) -> dict:
    # With frequency of 44100
    # Each frame is 0.0929 seconds with an overlap ratio of .5,
    # so moving over one frame moves 0.046 seconds
    # 1 second of frames is 21.55 frames.
    #
    # add option to specify which value to sort by?
    # PSNR

    t = time.time()

    img_width, overlap_ratio = get_frame_width_and_overlap(img_width, overlap_ratio)

    target_samples, _ = read(target_file_path)
    target_arr2d = fingerprint.fingerprint(target_samples, retspec=True)
    target_arr2d = target_arr2d[0 : -fingerprint.threshold]
    transposed_target_arr2d = np.clip(np.transpose(target_arr2d), 0, 255)

    against_samples, _ = read(against_file_path)
    against_arr2d = fingerprint.fingerprint(against_samples, retspec=True)
    against_arr2d = against_arr2d[0 : -fingerprint.threshold]
    transposed_against_arr2d = np.clip(np.transpose(against_arr2d), 0, 255)

    th, tw = transposed_target_arr2d.shape
    ah, aw = transposed_against_arr2d.shape

    # print(f"Target height: {th}, target width: {tw}")
    # print(f"against height: {ah}")
    # print(f"length of target: {len(transposed_target_arr2d)}")
    print(
        f"Comparing {os.path.basename(target_file_path)} against {os.path.basename(against_file_path)} ",
        end="",
    )

    # create index list
    index_list = []
    for i in range(0, th, overlap_ratio):
        for j in range(0, ah, overlap_ratio):
            if i + img_width < th and j + img_width < ah:
                index_list += [(i, j)]
    if th > overlap_ratio and ah > overlap_ratio:
        index_list += [(th - img_width - 1, ah - img_width - 1)]

    _calculate_comp_values = partial(
        calculate_comp_values,
        img_width=img_width,
        target_arr2d=transposed_target_arr2d,
        against_arr2d=transposed_against_arr2d,
        threshold=volume_threshold,
    )

    # calculate all mse and ssim values
    if use_multiprocessing == True:

        try:
            nprocesses = num_processes or multiprocessing.cpu_count()
        except NotImplementedError:
            nprocesses = 1
        else:
            nprocesses = 1 if nprocesses <= 0 else nprocesses

        with multiprocessing.Pool(nprocesses) as pool:
            results_list = pool.map(_calculate_comp_values, index_list)
            pool.close()
            pool.join()
    else:
        results_list = []
        for i in index_list:
            results_list += _calculate_comp_values(i)

    print(f"done")
    print("Calculating results... ", end="")

    if plot:
        plot_two_images(target_arr2d, against_arr2d)

    file_match = process_results(results_list, os.path.basename(against_file_path))

    print("done")

    t = time.time() - t

    result = {}

    result["match_time"] = t
    result["match_info"] = file_match

    return result


def visrecognize_directory(target_file_path: str, against_directory: str):
    results = {}
    return results


def process_results(results_list, filename):
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
        match_offsets, reverse=True, key=lambda x: x[0][1]
    )  # sort by ssim must be reversed for ssim

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

    return match


def plot_two_images(imageA, imageB, title="Comparison", mse=None, ssim_value=None):
    # setup the figure
    fig = plt.figure(title)
    if mse or ssim_value:
        plt.suptitle(f"MSE: {mse:.4f}, SSIM: {ssim_value:.4f}")
    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA)  # , cmap=plt.cm.gray)
    plt.gca().invert_yaxis()
    # plt.axis("off")
    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB)  # , cmap=plt.cm.gray)
    plt.gca().invert_yaxis()
    # plt.axis("off")
    # show the images
    plt.show()
