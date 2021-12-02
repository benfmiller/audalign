from audalign.config import BaseConfig
import typing


class VisrecognizeConfig(BaseConfig):
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

    # TODO handle this better
    img_width: float = 1.0
    volume_threshold: float = 215.0
    volume_floor: float = 10.0
    vert_scaling: float = 1.0
    horiz_scaling: float = 1.0
    calc_mse: bool = False

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