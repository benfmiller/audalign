"""Alignment isn't quite accurate enough for this yet"""


def rank_alignment(alignment):
    ranks = {}
    if "fine_match_info" in alignment.keys():
        ranks["fine_match_info"] = _rank_alignment(alignment["fine_match_info"])
    ranks["match_info"] = _rank_alignment(alignment=alignment["match_info"])
    return ranks


def _rank_alignment(alignment):
    # TODO
    ...


def rank_recognition(
    alignment,
):
    # TODO
    ...


def calc_technique(keys):
    if "confidence" in keys and "scaling_factor" not in keys:
        return "fingerprints"
    elif "confidence" not in keys:
        return "visual"
    elif "offset_frames" in keys:
        return "correlation_spectrogram"
    else:
        return "correlation"


def speed_of_sound(degrees: int) -> int:
    """degrees in celcius"""
    return 331.3 + (degrees * 0.606)


def event_a_is_closer(offset_a: int, offset_b: int) -> bool:
    return offset_a > offset_b


def distance_from_event():
    ...


def angle_two_events():
    ...


def which_is_first():
    ...
