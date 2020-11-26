from operator import index
import re
import cv2
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
    index_tuple, img_width=0, target_arr2d=[[]], against_arr2d=[[]]
):
    try:
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
    use_multiprocessing=True,
    plot=False,
) -> dict:
    # With frequency of 44100
    # Each frame is 0.0929 seconds with an overlap ratio of .5,
    # so moving over one frame moves 0.046 seconds
    # 1 second of frames is 21.55 frames.

    img_width, overlap_ratio = get_frame_width_and_overlap(img_width, overlap_ratio)

    results = {}
    target_samples, _ = read(target_file_path)
    target_arr2d = fingerprint.fingerprint(target_samples, retspec=True)
    transposed_target_arr2d = np.transpose(target_arr2d)

    against_samples, _ = read(against_file_path)
    against_arr2d = fingerprint.fingerprint(against_samples, retspec=True)
    transposed_against_arr2d = np.transpose(against_arr2d)

    # print(f"target max = {np.amax(target_arr2d)}")
    # print(f"against max = {np.amax(against_arr2d)}")

    th, tw = transposed_target_arr2d.shape
    ah, aw = transposed_against_arr2d.shape

    print(f"Target height: {th}, target width: {tw}")
    print(f"against height: {ah}")
    print(f"length of target: {len(transposed_target_arr2d)}")

    # create index list
    index_list = []
    for i in range(0, th, overlap_ratio):
        for j in range(0, ah, overlap_ratio):
            if i + img_width < th and j + img_width < ah:
                # print(f"{i}, {j}")
                index_list += [(i, j)]
    if th > overlap_ratio and ah > overlap_ratio:
        index_list += [(th - img_width - 1, ah - img_width - 1)]

    _calculate_comp_values = partial(
        calculate_comp_values,
        img_width=img_width,
        target_arr2d=transposed_target_arr2d,
        against_arr2d=transposed_against_arr2d,
    )

    # calculate all mse and ssim values
    if use_multiprocessing == True:

        try:
            nprocesses = multiprocessing.cpu_count()
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

    # results_list = [
    #     (
    #         x[0],
    #         x[1],
    #         calculate_comp_values(
    #             transposed_target_arr2d[x[0] : x[0] + img_width],
    #             transposed_against_arr2d[x[1] : x[1] + img_width],
    #         ),
    #     )
    #     for x in index_list
    # ]

    results_list = sorted(results_list, key=lambda x: x[2][0])

    if plot:
        plot_two_images(target_arr2d, against_arr2d)

    return results


def visrecognize_directory(target_file_path: str, against_directory: str):
    results = {}
    return results


"""
def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err
"""


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