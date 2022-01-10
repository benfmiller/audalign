from audalign.config import BaseConfig
import typing


class VisualConfig(BaseConfig):

    # width of spectrogram image for recognition in seconds.
    img_width: float = 1.0

    # doesn't find stats for sections with max volume below threshold.
    volume_threshold: float = 215.0

    # ignores volume levels below floor.
    volume_floor: float = 10.0

    # scales vertically to speed up calculations. Smaller numbers have smaller images.
    vert_scaling: float = 1.0

    # scales horizontally to speed up calculations. Smaller numbers have smaller images. Affects alignment granularity.
    horiz_scaling: float = 1.0

    # also calculates mean squared error for each shift if true. If false, uses default mse 20000000
    calc_mse: bool = False

    # cuts off top x rows returned from mlab.specgram in fingerprinter
    # Those high frequencies are very noisy and disrupt recognitions
    cutoff_top: int = 200

    freq_threshold: int = 100

    filter_matches = 1
    start_end_against: typing.Optional[tuple] = None

    CONFIDENCE = "ssim"

    ######################################################################
    # Sampling rate, related to the Nyquist conditions, which affects
    # the range frequencies we can detect.
    sample_rate = 44100

    ######################################################################
    # Size of the FFT window, affects frequency granularity
    # Which is 0.0929 seconds
    fft_window_size = 4096

    ######################################################################
    # Ratio by which each sequential window overlaps the last and the
    # next window. Higher overlap will allow a higher granularity of offset
    # matching, but potentially more fingerprints.
    DEFAULT_OVERLAP_RATIO = 0.5

    rankings_second_is_close_add: int = 0
    rankings_get_top_num_match = "num_matches"
    rankings_minus = ((0.98, 4), (0.96, 3), (0.94, 2), (0.91, 1), (0.0, 0))
    rankings_no_locality_top_match_tups = (
        (0.68, 10),
        (0.66, 9),
        (0.63, 8),
        (0.61, 7),
        (0.59, 6),
        (0.55, 5),
        (0.5, 4),
        (0, 1),
    )
    rankings_num_matches_tups = (
        (1, 9),
        (4, 8),
        (6, 7),
        (9, 6),
        (13, 4),
        (16, 3),
        (20, 2),
        (30, 1),
        (99999999999, 0),
    )