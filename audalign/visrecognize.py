import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

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


def calculate_comp_values(target_frame, against_frame):
    try:
        m = mean_squared_error(target_frame, against_frame)
        s = ssim(target_frame, against_frame)
        return (m, s)
    except ZeroDivisionError as e:
        m = 10000000
        print("zero division error")
        s = 10000000
        return (m, s)


def visrecognize(
    target_file_path: str,
    against_file_path: str,
    img_width=1.0,
    overlap_ratio=0.5,
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

    results_list = []

    # calculate all mse and ssim values
    for i in range(0, th - img_width, overlap_ratio):
        for j in range(0, ah - img_width, overlap_ratio):
            if i + img_width < tw and j + img_width < aw:
                # average signal power filter?
                results_list += [
                    (
                        i,
                        j,
                        calculate_comp_values(
                            transposed_target_arr2d[i : i + img_width],
                            transposed_against_arr2d[j : j + img_width],
                        ),
                    )
                ]

    # calculate mse and ssim for last frame
    if th > img_width and ah > img_width:
        # average signal power filter?
        results_list += [
            (
                th - img_width - 1,
                ah - img_width - 1,
                calculate_comp_values(
                    transposed_target_arr2d[th - img_width - 1 : th - 1],
                    transposed_against_arr2d[ah - img_width - 1 : ah - 1],
                ),
            )
        ]
    results_list = sorted(results_list, key=lambda x: x[2][0])
    print(results_list)

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