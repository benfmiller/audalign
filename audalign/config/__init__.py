import typing
from abc import ABC


class BaseConfig(ABC):
    passthrough_args = {}  # TODO incorporate this

    freq_threshold = 44100
    multiprocessing = True
    num_processors = None
    start_end = None
    plot = False
    max_lags: typing.Optional[float] = None
    filter_matches: typing.Optional[int] = None
    sample_rate = 44100
    match_len_filter: typing.Optional[int] = None
    locality = None  # Not all recognizers have to implement locality

    CONFIDENCE = "confidence"
    MATCH_TIME = "match_time"
    OFFSET_SAMPLES = "offset_frames"
    OFFSET_SECS = "offset_seconds"
    LOCALITY_FRAMES = "locality_frames"
    LOCALITY_SECS = "locality_seconds"

    rankings_no_locality_top_match_tups: tuple = ()
    rankings_locality_top_match_tups: tuple = ()
    rankings_minus: tuple = ()
    rankings_second_is_close_add: int = 1
    rankings_get_top_num_match: typing.Optional[str] = None
    rankings_num_matches_tups: typing.Optional[tuple] = None