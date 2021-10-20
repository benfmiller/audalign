import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import (
    generate_binary_structure,
    iterate_structure,
    binary_erosion,
)
import hashlib
from pydub.exceptions import CouldntDecodeError

np.seterr(divide="ignore")

IDX_FREQ_I = 0
IDX_TIME_J = 1

######################################################################
# Sampling rate, related to the Nyquist conditions, which affects
# the range frequencies we can detect.
DEFAULT_FS = 44100

######################################################################
# Size of the FFT window, affects frequency granularity
# Which is 0.0929 seconds
DEFAULT_WINDOW_SIZE = 4096

######################################################################
# Ratio by which each sequential window overlaps the last and the
# next window. Higher overlap will allow a higher granularity of offset
# matching, but potentially more fingerprints.
DEFAULT_OVERLAP_RATIO = 0.5

######################################################################
# Degree to which a fingerprint can be paired with its neighbors --
# higher will cause more fingerprints, but potentially better accuracy.
default_fan_value = 15

######################################################################
# Minimum amplitude in spectrogram in order to be considered a peak.
# This can be raised to reduce number of fingerprints, but can negatively
# affect accuracy.
# 50 roughly cuts number of fingerprints in half compared to 0
default_amp_min = 65

######################################################################
# Number of cells around an amplitude peak in the spectrogram in order
# for audalign to consider it a spectral peak. Higher values mean less
# fingerprints and faster matching, but can potentially affect accuracy.
peak_neighborhood_size = 20

######################################################################
# Thresholds on how close or far fingerprints can be in time in order
# to be paired as a fingerprint. If your max is too low, higher values of
# default_fan_value may not perform as expected.
min_hash_time_delta = 10
max_hash_time_delta = 200

######################################################################
# If True, will sort peaks temporally for fingerprinting;
# not sorting will cut down number of fingerprints, but potentially
# affect performance.
peak_sort = True

######################################################################
# Number of bits to grab from the front of the SHA1 hash in the
# fingerprint calculation. The more you grab, the more memory storage,
# with potentially lesser collisions of matches.
FINGERPRINT_REDUCTION = 20


threshold = 200


def _fingerprint_worker(
    file_path: str,
    hash_style="panako_mod",
    start_end: tuple = None,
    plot=False,
    accuracy=2,
    freq_threshold=200,
) -> tuple:
    import audalign
    import os

    """
    Runs the file through the fingerprinter and returns file_name and hashes

    Args
        file_path (str): file_path to be fingerprinted
        hash_style (str): which hash style to use : ['base','panako_mod','panako', 'base_three']
        start_end (tuple(float, float), optional): Silences before and after start and end. (0, -1) Silences last second, (5.4, 0) silences first 5.4 seconds
        plot (bool): displays the plot of the peaks if true
        accuracy (int): which accuracy level 1-4
        freq_threshold (int): what the freq threshold is in specgram bins

    Returns
    -------
        file_name (str, hashes : dict{str: [int]}): file_name and hash dictionary
    """

    audalign.Audalign._set_accuracy(accuracy)
    audalign.Audalign._set_freq_threshold(freq_threshold)

    if type(file_path) == str:
        file_name = os.path.basename(file_path)

        try:
            channel, _ = audalign.filehandler.read(file_path, start_end=start_end)
        except FileNotFoundError:
            print(f'"{file_path}" not found')
            return None, None
        except (
            CouldntDecodeError,
            IndexError,
        ):  # Pydub throws IndexErrors for some files on Ubuntu (json, txt, others?)
            print(f'File "{file_name}" could not be decoded')
            return None, None
    elif type(file_path) == tuple:
        from audalign.filehandler import get_shifted_file

        file_name = os.path.basename(file_path[0])
        channel = get_shifted_file(file_path[0], file_path[1], sample_rate=DEFAULT_FS)

    print(f"Fingerprinting {file_name}")
    hashes = fingerprint(
        channel,
        hash_style=hash_style,
        plot=plot,
    )

    print(f"Finished fingerprinting {file_name}")

    return file_name, hashes


def fingerprint(
    channel_samples,
    fs=DEFAULT_FS,
    wsize=DEFAULT_WINDOW_SIZE,
    wratio=DEFAULT_OVERLAP_RATIO,
    plot=False,
    hash_style="panako",
    retspec=False,
):
    """
    FFT the channel, log transform output, find local maxima, then return
    locally sensitive hashes.

    Args
        channel_samples (array[int]): audio file data
        fs (int): Sample Rate


    Returns
    -------
        hashes (dict{str: [int]}): hashes of the form dict{hash: location}
    """
    # FFT the signal and extract frequency components
    # To get the frequencies of each row, get the second returned component
    arr2D = mlab.specgram(
        channel_samples,
        NFFT=wsize,
        Fs=fs,
        window=mlab.window_hanning,
        noverlap=int(wsize * wratio),
    )[0]

    # apply log transform since specgram() returns linear array
    # print(arr2D)
    # print(max(arr2D[0]))
    arr2D = 10 * np.log2(arr2D)
    # TODO test this to see if we should change it back, would have to recalc default settings
    # arr2D = 10 * np.log10(arr2D, out=np.zeros_like(arr2D), where=(arr2D != 0))
    arr2D[arr2D == -np.inf] = 0  # replace infs with zeros
    # print(max(arr2D[0]))
    # print(f"length of arr2d {len(arr2D)}")
    # print(f"length of arr2d height {len(arr2D[1])}")

    if retspec:
        return arr2D

    # find local maxima
    local_maxima = get_2D_peaks(arr2D, plot=plot)

    # return hashes
    return generate_hashes(
        local_maxima,
        hash_style,
    )


def get_2D_peaks(arr2D, plot=False):
    #  http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.iterate_structure.html#scipy.ndimage.iterate_structure
    struct = generate_binary_structure(
        2, 1
    )  # 2 is faster here for connectivity, mainly saves time in maximum filter function.
    # 2 results in slightly less fingerprints (4/5?), which specifically could help with false detections in noise.
    # It would also lessen fingerprints at edges of sound events.
    # I think it's more important to keep those edges of sound events than worry about noise here or speed
    neighborhood = iterate_structure(struct, peak_neighborhood_size)

    # find local maxima using our filter shape
    local_max = maximum_filter(arr2D, footprint=neighborhood) == arr2D
    background = arr2D == 0
    eroded_background = binary_erosion(
        background, structure=neighborhood, border_value=1
    )

    # Boolean mask of arr2D with True at peaks (Fixed deprecated boolean operator by changing '-' to '^')
    detected_peaks = local_max ^ eroded_background

    # extract peaks
    amps = arr2D[detected_peaks]
    j, i = np.where(detected_peaks)

    # filter peaks
    amps = amps.flatten()
    peaks = zip(i, j, amps)
    peaks_filtered = filter(
        lambda x: x[2] > default_amp_min and x[1] > threshold, peaks
    )  # time, freq, amp
    # get indices for frequency and time
    frequency_idx = []
    time_idx = []
    for x in peaks_filtered:
        frequency_idx.append(x[1])
        time_idx.append(x[0])

    if plot:
        # scatter of the peaks
        fig, ax = plt.subplots()
        ax.imshow(arr2D)
        ax.scatter(time_idx, frequency_idx, color="r")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        ax.set_title("Spectrogram")
        plt.gca().invert_yaxis()
        plt.show()

    return zip(frequency_idx, time_idx)


def generate_hashes(peaks, hash_style):
    """
    Hash list structure:
       sha1_hash[0:30]    time_offset
    [(e05b341a9b77a51fd26..., 32), ... ]
    """
    peaks = list(peaks)
    if peak_sort:
        peaks = sorted(peaks, key=lambda x: x[1])
    # print("Length of Peaks List is: {}".format(len(peaks)))

    if hash_style == "panako_mod":
        return panako_mod(peaks)
    elif hash_style == "base":
        return base(peaks)
    elif hash_style == "panako":
        return panako(peaks)
    elif hash_style == "base_three":
        return base_three(peaks)
    else:
        print(f'Hash style "{hash_style}" is not inplemented')


def panako_mod(peaks):
    hash_dict = {}
    for i in range(0, len(peaks), 1):
        freq1 = peaks[i][IDX_FREQ_I]
        t1 = peaks[i][IDX_TIME_J]
        for j in range(1, default_fan_value - 1):
            if i + j < len(peaks):
                freq2 = peaks[i + j][IDX_FREQ_I]
                t2 = peaks[i + j][IDX_TIME_J]
                for k in range(j + 1, default_fan_value):
                    if (i + k) < len(peaks):

                        freq3 = peaks[i + k][IDX_FREQ_I]
                        t3 = peaks[i + k][IDX_TIME_J]

                        t_delta = t3 - t1

                        if (
                            t_delta >= min_hash_time_delta
                            and t_delta <= max_hash_time_delta
                        ):

                            t_delta = t2 - t1

                            if (
                                t_delta >= min_hash_time_delta
                                and t_delta <= max_hash_time_delta
                            ):
                                h = hashlib.sha1(
                                    f"{freq1-freq2}|{freq2-freq3}|{(t2-t1)/(t3-t1):.8f}".encode(
                                        "utf-8"
                                    )
                                ).hexdigest()[0:FINGERPRINT_REDUCTION]
                                if h not in hash_dict:
                                    hash_dict[h] = [int(t1)]
                                else:
                                    hash_dict[h] += [int(t1)]
    return hash_dict


def base(peaks):
    hash_dict = {}
    for i in range(0, len(peaks), 1):
        freq1 = peaks[i][IDX_FREQ_I]
        t1 = peaks[i][IDX_TIME_J]
        for j in range(1, default_fan_value):
            if i + j < len(peaks):
                freq2 = peaks[i + j][IDX_FREQ_I]
                t2 = peaks[i + j][IDX_TIME_J]
                t_delta = t2 - t1

                if t_delta >= min_hash_time_delta and t_delta <= max_hash_time_delta:

                    h = hashlib.sha1(
                        f"{freq1}|{freq2}|{t_delta}".encode("utf-8")
                    ).hexdigest()[0:FINGERPRINT_REDUCTION]
                    if h not in hash_dict:
                        hash_dict[h] = [int(t1)]
                    else:
                        hash_dict[h] += [int(t1)]
    return hash_dict


def panako(peaks):
    hash_dict = {}
    for i in range(0, len(peaks), 1):
        freq1 = peaks[i][IDX_FREQ_I]
        t1 = peaks[i][IDX_TIME_J]
        for j in range(1, default_fan_value - 1):
            if i + j < len(peaks):
                freq2 = peaks[i + j][IDX_FREQ_I]
                t2 = peaks[i + j][IDX_TIME_J]
                for k in range(j + 1, default_fan_value):
                    if (i + k) < len(peaks):

                        freq3 = peaks[i + k][IDX_FREQ_I]
                        t3 = peaks[i + k][IDX_TIME_J]

                        t_delta1 = t3 - t1

                        if (
                            t_delta1 >= min_hash_time_delta
                            and t_delta1 <= max_hash_time_delta
                        ):

                            t_delta2 = t2 - t1

                            if (
                                t_delta2 >= min_hash_time_delta
                                and t_delta2 <= max_hash_time_delta
                            ):
                                h = hashlib.sha1(
                                    f"{freq1-freq2}|{freq2-freq3}|{freq1//400}|{freq3//400}|{(t_delta2)/(t_delta1):.8f}".encode(
                                        "utf-8"
                                    )
                                ).hexdigest()[0:FINGERPRINT_REDUCTION]
                                if h not in hash_dict:
                                    hash_dict[h] = [int(t1)]
                                else:
                                    hash_dict[h] += [int(t1)]
    return hash_dict


def base_three(peaks):
    hash_dict = {}
    for i in range(0, len(peaks), 1):
        freq1 = peaks[i][IDX_FREQ_I]
        t1 = peaks[i][IDX_TIME_J]
        for j in range(1, default_fan_value - 1):
            if i + j < len(peaks):
                freq2 = peaks[i + j][IDX_FREQ_I]
                t2 = peaks[i + j][IDX_TIME_J]
                for k in range(j + 1, default_fan_value):
                    if (i + k) < len(peaks):

                        freq3 = peaks[i + k][IDX_FREQ_I]
                        t3 = peaks[i + k][IDX_TIME_J]

                        t_delta1 = t3 - t1

                        if (
                            t_delta1 >= min_hash_time_delta
                            and t_delta1 <= max_hash_time_delta
                        ):

                            t_delta2 = t2 - t1

                            if (
                                t_delta2 >= min_hash_time_delta
                                and t_delta2 <= max_hash_time_delta
                            ):
                                h = hashlib.sha1(
                                    f"{freq1}|{freq2}|{freq3}|{t_delta1}|{t_delta2}".encode(
                                        "utf-8"
                                    )
                                ).hexdigest()[0:FINGERPRINT_REDUCTION]
                                if h not in hash_dict:
                                    hash_dict[h] = [int(t1)]
                                else:
                                    hash_dict[h] += [int(t1)]
    return hash_dict
