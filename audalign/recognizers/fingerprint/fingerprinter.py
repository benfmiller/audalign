import hashlib

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from audalign.config.fingerprint import FingerprintConfig
from pydub.exceptions import CouldntDecodeError
from scipy.ndimage import (binary_erosion, generate_binary_structure,
                           iterate_structure, maximum_filter)

np.seterr(divide="ignore")


def _fingerprint_worker(
    file_path: str,
    config: FingerprintConfig,
) -> tuple:
    import os

    import audalign

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
    if type(file_path) == str:
        file_name = os.path.basename(file_path)

        try:
            channel, _ = audalign.filehandler.read(
                file_path, start_end=config.start_end, sample_rate=config.sample_rate
            )
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
        channel = get_shifted_file(
            file_path[0], file_path[1], sample_rate=config.sample_rate
        )

    print(f"Fingerprinting {file_name}")
    hashes = fingerprint(
        channel,
        config=config,
    )

    print(f"Finished fingerprinting {file_name}")

    return file_name, hashes


def fingerprint(
    channel_samples,
    config: FingerprintConfig,
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
    arr2D, frequencies, _ = mlab.specgram(
        channel_samples,
        NFFT=config.fft_window_size,
        Fs=config.sample_rate,
        window=mlab.window_hanning,
        noverlap=int(config.fft_window_size * config.DEFAULT_OVERLAP_RATIO),
    )
    # )[0]

    # apply log transform since specgram() returns linear array
    # print(arr2D)
    # print(max(arr2D[0]))
    arr2D = 10 * np.log2(arr2D)
    # got better results with a log2, but this means that nothing is in terms of decibels
    # arr2D = 10 * np.log10(arr2D, out=np.zeros_like(arr2D), where=(arr2D != 0))
    arr2D[arr2D == -np.inf] = 0  # replace infs with zeros
    # print(max(arr2D[0]))
    # print(f"length of arr2d {len(arr2D)}")
    # print(f"length of arr2d height {len(arr2D[1])}")

    if config.freq_threshold is not None:
        index = 0
        for i, frequency in enumerate(frequencies):
            if frequency > config.freq_threshold - 0.0001:
                index = i
                break
        else:
            index = len(frequencies)
        arr2D[0:index] = 0

    if retspec:
        return arr2D

    # find local maxima
    local_maxima = get_2D_peaks(arr2D, config=config)

    # return hashes
    return generate_hashes(
        local_maxima,
        config,
    )


def get_2D_peaks(arr2D, config: FingerprintConfig):
    #  http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.iterate_structure.html#scipy.ndimage.iterate_structure
    struct = generate_binary_structure(
        2, 1
    )  # 2 is faster here for connectivity, mainly saves time in maximum filter function.
    # 2 results in slightly less fingerprints (4/5?), which specifically could help with false detections in noise.
    # It would also lessen fingerprints at edges of sound events.
    # I think it's more important to keep those edges of sound events than worry about noise here or speed
    neighborhood = iterate_structure(struct, config.peak_neighborhood_size)

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

    # This cuts off by frequency band rather than hertz
    # peaks_filtered = filter(
    #     lambda x: x[2] > config.default_amp_min and x[1] > config.freq_threshold, peaks
    # )  # time, freq, amp
    peaks_filtered = filter(
        lambda x: x[2] > config.default_amp_min, peaks
    )  # time, freq, amp
    # get indices for frequency and time
    frequency_idx = []
    time_idx = []
    for x in peaks_filtered:
        frequency_idx.append(x[1])
        time_idx.append(x[0])

    if config.plot:
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


def generate_hashes(peaks, config: FingerprintConfig):
    """
    Hash list structure:
       sha1_hash[0:30]    time_offset
    [(e05b341a9b77a51fd26..., 32), ... ]
    """
    peaks = list(peaks)
    if config.peak_sort:
        peaks = sorted(peaks, key=lambda x: x[1])
    # print("Length of Peaks List is: {}".format(len(peaks)))

    if config.hash_style == "panako_mod":
        return panako_mod(peaks, config=config)
    elif config.hash_style == "base":
        return base(peaks, config=config)
    elif config.hash_style == "panako":
        return panako(peaks, config=config)
    elif config.hash_style == "base_three":
        return base_three(peaks, config=config)
    else:
        print(f'Hash style "{config.hash_style}" is not inplemented')


def panako_mod(peaks, config: FingerprintConfig):
    hash_dict = {}
    for i in range(0, len(peaks), 1):
        freq1 = peaks[i][config._IDX_FREQ_I]
        t1 = peaks[i][config._IDX_TIME_J]
        for j in range(1, config.default_fan_value - 1):
            if i + j < len(peaks):
                freq2 = peaks[i + j][config._IDX_FREQ_I]
                t2 = peaks[i + j][config._IDX_TIME_J]
                for k in range(j + 1, config.default_fan_value):
                    if (i + k) < len(peaks):

                        freq3 = peaks[i + k][config._IDX_FREQ_I]
                        t3 = peaks[i + k][config._IDX_TIME_J]

                        t_delta = t3 - t1

                        if (
                            t_delta >= config.min_hash_time_delta
                            and t_delta <= config.max_hash_time_delta
                        ):

                            t_delta = t2 - t1

                            if (
                                t_delta >= config.min_hash_time_delta
                                and t_delta <= config.max_hash_time_delta
                            ):
                                h = hashlib.sha1(
                                    f"{freq1-freq2}|{freq2-freq3}|{(t2-t1)/(t3-t1):.8f}".encode(
                                        "utf-8"
                                    )
                                ).hexdigest()[0 : config.FINGERPRINT_REDUCTION]
                                if h not in hash_dict:
                                    hash_dict[h] = [int(t1)]
                                else:
                                    hash_dict[h] += [int(t1)]
    return hash_dict


def base(peaks, config: FingerprintConfig):
    hash_dict = {}
    for i in range(0, len(peaks), 1):
        freq1 = peaks[i][config._IDX_FREQ_I]
        t1 = peaks[i][config._IDX_TIME_J]
        for j in range(1, config.default_fan_value):
            if i + j < len(peaks):
                freq2 = peaks[i + j][config._IDX_FREQ_I]
                t2 = peaks[i + j][config._IDX_TIME_J]
                t_delta = t2 - t1

                if (
                    t_delta >= config.min_hash_time_delta
                    and t_delta <= config.max_hash_time_delta
                ):

                    h = hashlib.sha1(
                        f"{freq1}|{freq2}|{t_delta}".encode("utf-8")
                    ).hexdigest()[0 : config.FINGERPRINT_REDUCTION]
                    if h not in hash_dict:
                        hash_dict[h] = [int(t1)]
                    else:
                        hash_dict[h] += [int(t1)]
    return hash_dict


def panako(peaks, config: FingerprintConfig):
    hash_dict = {}
    for i in range(0, len(peaks), 1):
        freq1 = peaks[i][config._IDX_FREQ_I]
        t1 = peaks[i][config._IDX_TIME_J]
        for j in range(1, config.default_fan_value - 1):
            if i + j < len(peaks):
                freq2 = peaks[i + j][config._IDX_FREQ_I]
                t2 = peaks[i + j][config._IDX_TIME_J]
                for k in range(j + 1, config.default_fan_value):
                    if (i + k) < len(peaks):

                        freq3 = peaks[i + k][config._IDX_FREQ_I]
                        t3 = peaks[i + k][config._IDX_TIME_J]

                        t_delta1 = t3 - t1

                        if (
                            t_delta1 >= config.min_hash_time_delta
                            and t_delta1 <= config.max_hash_time_delta
                        ):

                            t_delta2 = t2 - t1

                            if (
                                t_delta2 >= config.min_hash_time_delta
                                and t_delta2 <= config.max_hash_time_delta
                            ):
                                h = hashlib.sha1(
                                    f"{freq1-freq2}|{freq2-freq3}|{freq1//400}|{freq3//400}|{(t_delta2)/(t_delta1):.8f}".encode(
                                        "utf-8"
                                    )
                                ).hexdigest()[0 : config.FINGERPRINT_REDUCTION]
                                if h not in hash_dict:
                                    hash_dict[h] = [int(t1)]
                                else:
                                    hash_dict[h] += [int(t1)]
    return hash_dict


def base_three(peaks, config: FingerprintConfig):
    hash_dict = {}
    for i in range(0, len(peaks), 1):
        freq1 = peaks[i][config._IDX_FREQ_I]
        t1 = peaks[i][config._IDX_TIME_J]
        for j in range(1, config.default_fan_value - 1):
            if i + j < len(peaks):
                freq2 = peaks[i + j][config._IDX_FREQ_I]
                t2 = peaks[i + j][config._IDX_TIME_J]
                for k in range(j + 1, config.default_fan_value):
                    if (i + k) < len(peaks):

                        freq3 = peaks[i + k][config._IDX_FREQ_I]
                        t3 = peaks[i + k][config._IDX_TIME_J]

                        t_delta1 = t3 - t1

                        if (
                            t_delta1 >= config.min_hash_time_delta
                            and t_delta1 <= config.max_hash_time_delta
                        ):

                            t_delta2 = t2 - t1

                            if (
                                t_delta2 >= config.min_hash_time_delta
                                and t_delta2 <= config.max_hash_time_delta
                            ):
                                h = hashlib.sha1(
                                    f"{freq1}|{freq2}|{freq3}|{t_delta1}|{t_delta2}".encode(
                                        "utf-8"
                                    )
                                ).hexdigest()[0 : config.FINGERPRINT_REDUCTION]
                                if h not in hash_dict:
                                    hash_dict[h] = [int(t1)]
                                else:
                                    hash_dict[h] += [int(t1)]
    return hash_dict
