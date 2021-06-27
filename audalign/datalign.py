import numpy as np


def rank_alignment(alignment):
    ranks = {}
    if "fine_match_info" in alignment.keys():
        ranks["fine_match_info"] = _rank_alignment(alignment["fine_match_info"])
    ranks["match_info"] = _rank_alignment(alignment=alignment["match_info"])
    return ranks


def _rank_alignment(alignment):
    new_ranks = {}
    if "match_info" in alignment.keys():
        alignment = alignment["match_info"]
    if "offset_seconds" in alignment.keys():
        return rank_recognition(alignment=alignment)
    else:
        for key in alignment.keys():
            new_ranks[key] = _rank_alignment(alignment=alignment[key])
    return new_ranks


def rank_recognition(
    alignment,
):
    """Big decision tree"""
    # TODO
    offset_seconds = alignment["offset_seconds"]
    if "confidence" in alignment.keys() and "scaling_factor" not in alignment.keys():
        # fingerprints
        rank_minus = ((0.95, 4), (0.9, 3), (0.85, 2), (0.8, 1), (0.0, 0))
        confidences = alignment["confidence"]
        if alignment["locality_seconds"][0] is not None:  # locality
            ...
            return 0
        else:  # no locality
            top_match_tups = (
                (500, 10),
                (200, 9),
                (100, 8),
                (70, 7),
                (30, 6),
                (15, 5),
                (10, 4),
            )
        rank = _calc_rank(
            confidences=confidences,
            top_match_tups=top_match_tups,
            rank_minus=rank_minus,
        )
        if abs(offset_seconds[0] - offset_seconds[1]) < 0.5:
            # second best is very close means probably very close
            rank += 1
    elif "confidence" not in alignment.keys():
        # visual
        return 0
    elif "offset_frames" in alignment.keys():
        # correlation_spectrogram
        if alignment["locality_seconds"][0] is not None:
            ...
        else:
            ...
        return 0
    else:
        # correlation
        if alignment["locality_seconds"][0] is not None:
            ...
        else:
            ...
        return 0
    return int(np.clip(rank, 1, 10))


def _calc_rank(confidences, top_match_tups, rank_minus):
    top_match = confidences[0]
    second_match = confidences[1] if len(confidences) > 1 else 0
    rank = [top_match > x[0] for x in top_match_tups]
    rank = top_match_tups[rank.index(True)][1]
    second_match_list = [second_match >= top_match * x[0] for x in rank_minus]
    rank -= rank_minus[second_match_list.index(True)][1]


def calc_technique(keys: dict.keys):
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
