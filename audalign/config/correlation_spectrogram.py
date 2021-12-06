from audalign.config import BaseConfig
import typing


class CorrelationSpectrogramConfig(BaseConfig):
    """
    hash style has four options. All fingerprints must be of the same hash style to match.

    'base' hash style consists of two peaks. Two frequencies and a time difference.
    Creates many matches but is insensitive to noise.

    'panako' hash style consists of three peaks. Two differences in frequency, two frequency
    bands, one time difference ratio. Creates few matches, very resistant to noise.

    'panako_mod' hash style consists of three peaks. Two differences in frequency and one
    time difference ratio. Creates less matches than base, more than panako. moderately
    resistant to noise

    'base_three' hash style consists of three peaks. Three frequencies and two time differences.

    multiprocessing is set to True by default

    There are four accuracy levels with 1 being the lowest accuracy but the fastest. 3 is the highest recommended.
    4 gives the highest accuracy, but can take several gigabytes of memory for a couple files.
    Accuracy settings are acheived by manipulations in fingerprinting variables.
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
