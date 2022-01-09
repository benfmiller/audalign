import typing
from abc import ABC


class BaseConfig(ABC):

    # some recogniizers pass args to internal functions
    passthrough_args = {}

    freq_threshold = 200
    multiprocessing = True
    num_processors = None

    # start_end (tuple(float, float), optional): Silences before and after start and end.
    # (0, -1) Silences last second, (5.4, 0) silences first 5.4 seconds
    start_end = None

    # Whether or not to plot during recognitions
    plot = False

    # Maximum lags in seconds.
    max_lags: typing.Optional[float] = None

    # Filters based on confidence.
    filter_matches: typing.Optional[int] = None

    # Limits number of matches returned. Defaults to 30.
    match_len_filter: typing.Optional[int] = None

    # Filters matches to only count within locality. In seconds
    # Not all recognizers have to implement locality
    locality: typing.Optional[float] = None

    # Decodes audio file to this sample rate
    sample_rate = 44100

    # keys in results dictionaries
    CONFIDENCE = "confidence"
    MATCH_TIME = "match_time"
    OFFSET_SAMPLES = "offset_frames"
    OFFSET_SECS = "offset_seconds"
    LOCALITY_FRAMES = "locality_frames"
    LOCALITY_SECS = "locality_seconds"

    ######################################################################

    # Add to ranking if second match is close
    rankings_second_is_close_add: int = 1

    # If not using locality, tups of form (confidence, rank value) where confidence above
    # that tuples confidence value gets that ranking
    rankings_no_locality_top_match_tups: tuple = ()

    # If using locality, tups of form (confidence, rank value) where confidence above
    # that tuples confidence value gets that ranking
    rankings_locality_top_match_tups: tuple = ()

    # Subtracts from ranking if second closest confidence is within value
    # of the form (proportion threshold, subtract from ranking)
    rankings_minus: tuple = ()

    # if not None, gets the number of matches (used in visual)
    rankings_get_top_num_match: typing.Optional[str] = None

    # used if rankings_get_top_num_match is not None. (used in visual)
    # subtracts second value from ranking if num matches is above first value
    rankings_num_matches_tups: typing.Optional[tuple] = None