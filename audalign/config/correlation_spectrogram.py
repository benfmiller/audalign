from audalign.config import BaseConfig
import typing


class CorrelationSpectrogramConfig(BaseConfig):
    """
    multiprocessing is set to True by default
    """

    set_parameters = BaseConfig.set_parameters
    set_parameters.update(
        "filter_matches",
        "locality",
        "locality_filter_prop",
        "extensions",
        "start_end_agaisnt",
    )
    filter_matches = 1
    locality: typing.Optional[float] = None
    locality_filter_prop: typing.Optional[float] = None
    start_end_against: typing.Optional[tuple] = None
    extensions = ["*"]
    _IDX_FREQ_I = 0
    _IDX_TIME_J = 1

    ######################################################################
    # Sampling rate, related to the Nyquist conditions, which affects
    # the range frequencies we can detect.
    sample_rate = 44100

    ######################################################################
    # Size of the FFT window, affects frequency granularity
    # Which is 0.0929 seconds
    fft_window_size = 4096
    set_parameters.add("fft_window_size")

    ######################################################################
    # Ratio by which each sequential window overlaps the last and the
    # next window. Higher overlap will allow a higher granularity of offset
    # matching, but potentially more fingerprints.
    DEFAULT_OVERLAP_RATIO = 0.5
    set_parameters.add("DEFAULT_OVERLAP_RATIO")

    freq_threshold = 200
    set_parameters.add("freq_threshold")

    rankings_no_locality_top_match_tups = (
        (4.5, 10),
        (4, 9),
        (3.5, 8),
        (2.9, 7),
        (2.2, 6),
        (1.5, 5),
        (1, 4),
        (0.7, 3),
        (0.4, 2),
        (0, 1),
    )
    rankings_locality_top_match_tups = (
        (9, 10),
        (7.5, 9),
        (6, 8),
        (4, 7),
        (3, 6),
        (2, 5),
        (1, 4),
        (0, 1),
    )
    rankings_minus = (
        (0.96, 4),
        (0.92, 3),
        (0.89, 2),
        (0.85, 1),
        (0.8, 0),
        (0.75, -1),
        (0.7, -2),
        (0.65, -3),
        (0.1, -4),
        (0.0, 0),
    )
