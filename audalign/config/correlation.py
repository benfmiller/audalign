from audalign.config import BaseConfig
import typing


class CorrelationConfig(BaseConfig):
    """
    multiprocessing is set to True by default

    There are four accuracy levels with 1 being the lowest accuracy but the fastest. 3 is the highest recommended.
    4 gives the highest accuracy, but can take several gigabytes of memory for a couple files.
    Accuracy settings are acheived by manipulations in fingerprinting variables.
    """

    # filter_matches (float): Filters based on confidence. Ranges between 0 and 1. Defaults to 0
    filter_matches = 0.0

    # Within each offset, filters locality tuples by proportion of highest confidence to tuple confidence
    locality_filter_prop: typing.Optional[float] = None
    start_end_against: typing.Optional[tuple] = None
    extensions = ["*"]
    _IDX_FREQ_I = 0
    _IDX_TIME_J = 1

    ######################################################################
    # Sampling rate, related to the Nyquist conditions, which affects
    # the range frequencies we can detect.
    sample_rate = 8000

    ######################################################################
    # Size of the FFT window, affects frequency granularity
    # Which is 0.0929 seconds
    fft_window_size = 4096

    ######################################################################
    # Ratio by which each sequential window overlaps the last and the
    # next window. Higher overlap will allow a higher granularity of offset
    # matching, but potentially more fingerprints.
    DEFAULT_OVERLAP_RATIO = 0.5

    SCALING_16_BIT = 65536
    LOCALITY_OVERLAP_RATIO = 0.5
    DEFAULT_LOCALITY_FILTER_PROP = 0.6

    rankings_no_locality_top_match_tups = (
        (8, 10),
        (6, 9),
        (4, 8),
        (3, 7),
        (2, 6),
        (1.5, 5),
        (1, 4),
        (0, 1),
    )
    rankings_locality_top_match_tups = (
        (15, 10),
        (12, 9),
        (9, 8),
        (7, 7),
        (5.5, 6),
        (3, 5),
        (2, 4),
        (0, 1),
    )
    rankings_minus = (
        (0.95, 4),
        (0.9, 3),
        (0.85, 1),
        (0.8, 0),
        (0.75, -1),
        (0.7, -2),
        (0.65, -3),
        (0.1, -4),
        (0.0, 0),
    )