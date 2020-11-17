import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

from audalign.fingerprint import fingerprint
from audalign.filehandler import read


def visrecognize(target_file_path: str, against_file_path: str):
    # With frequency of 44100
    # Each frame is 0.0929 seconds with an overlap ratio of .5,
    # so moving over one frame moves 0.046 seconds
    # 1 second of frames is 21.55 frames.

    results = {}
    target_samples, _ = read(target_file_path)
    target_arr2d = fingerprint(target_samples, retspec=True)
    transposed_target_arr2d = np.transpose(target_arr2d)

    against_samples, _ = read(against_file_path)
    against_arr2d = fingerprint(against_samples, retspec=True)
    transposed_against_arr2d = np.transpose(against_arr2d)

    # print(f"target max = {np.amax(target_arr2d)}")
    # print(f"against max = {np.amax(against_arr2d)}")

    th, tw = target_arr2d.shape
    ah, aw = against_arr2d.shape

    print(f"Target height: {th}, target width: {tw}")
    print(f"against height: {ah}")
    print(f"length of target: {len(target_arr2d)}")

    m = mean_squared_error(target_arr2d[0:4000], against_arr2d[0:4000])
    s = ssim(target_arr2d[0:4000], against_arr2d[0:4000])

    fig = plt.figure("Test")
    # show first image
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(target_arr2d)  # , cmap=plt.cm.gray) for gray colors
    # plt.axis("off")
    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(against_arr2d)  # , cmap=plt.cm.gray) for gray colors
    # plt.axis("off")
    # show the images
    plt.show()

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


def plot_two_images(imageA, imageB, title="comparison", mse=None, ssim_value=None):
    # compute the mean squared error and structural similarity
    # index for the images
    m, s = 0, 0
    # m = mse(imageA, imageB)
    # s = ssim(imageA, imageB)
    # setup the figure
    fig = plt.figure(title)
    if mse and ssim_value:
        plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA)  # , cmap=plt.cm.gray)
    # plt.axis("off")
    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB)  # , cmap=plt.cm.gray)
    # plt.axis("off")
    # show the images
    plt.show()


def included():
    # load the images -- the original, the original + contrast,
    # and the original + photoshop
    original = cv2.imread("images/jp_gates_original.png")
    contrast = cv2.imread("images/jp_gates_contrast.png")
    shopped = cv2.imread("images/jp_gates_photoshopped.png")
    # convert the images to grayscale
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
    shopped = cv2.cvtColor(shopped, cv2.COLOR_BGR2GRAY)

    # initialize the figure
    fig = plt.figure("Images")
    images = ("Original", original), ("Contrast", contrast), ("Photoshopped", shopped)
    # loop over the images
    for (i, (name, image)) in enumerate(images):
        # show the image
        ax = fig.add_subplot(1, 3, i + 1)
        ax.set_title(name)
        plt.imshow(image, cmap=plt.cm.gray)
        plt.axis("off")
    # show the figure
    plt.show()
    # compare the images
    plot_two_images(original, original, "Original vs. Original")
    plot_two_images(original, contrast, "Original vs. Contrast")
    plot_two_images(original, shopped, "Original vs. Photoshopped")
