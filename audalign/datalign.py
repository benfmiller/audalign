import numpy as np

from audalign.recognizers import BaseRecognizer


def rank_alignment(alignment, recognizer: BaseRecognizer):
    ranks = {}
    if "fine_match_info" in alignment.keys():
        ranks = _rank_alignment(alignment["fine_match_info"], recognizer)
    else:
        ranks = _rank_alignment(
            alignment=alignment["match_info"], recognizer=recognizer
        )
    return ranks


def _rank_alignment(alignment: dict, recognizer: BaseRecognizer):
    new_ranks = {}
    if alignment is None:
        return 0
    if "match_info" in alignment.keys():
        alignment = alignment["match_info"]
    if "offset_seconds" in alignment.keys():
        return rank_recognition(alignment=alignment, recognizer=recognizer)
    else:
        for key in alignment.keys():
            new_ranks[key] = _rank_alignment(
                alignment=alignment[key], recognizer=recognizer
            )
    return new_ranks


def rank_recognition(alignment, recognizer: BaseRecognizer):
    """Big decision tree

    Locality is tricky because the locality width affect confidence so much.
    The current values reflect an average locality value (20ish to 3 ish).
    Should still be taken with a grain of salt.
    """
    offset_seconds = alignment["offset_seconds"]
    confidences = alignment[recognizer.config.CONFIDENCE]
    top_match_tups = recognizer.config.rankings_no_locality_top_match_tups
    if recognizer.config.locality is not None:
        top_match_tups = recognizer.config.rankings_locality_top_match_tups
    if alignment.get("scaling_factor") is not None:
        confidences = [x * alignment["scaling_factor"] for x in confidences]

    num_matches_tups = recognizer.config.rankings_num_matches_tups
    top_num_match = None
    if recognizer.config.rankings_get_top_num_match is not None:
        top_num_match = alignment[recognizer.config.rankings_get_top_num_match][0]
    rank = _calc_rank(
        confidences=confidences,
        top_match_tups=top_match_tups,
        rank_minus=recognizer.config.rankings_minus,
        offset_seconds=offset_seconds,
        num_matches_tups=num_matches_tups,
        num_match=top_num_match,
    )
    if len(offset_seconds) > 1 and abs(offset_seconds[0] - offset_seconds[1]) < 0.5:
        # second best is very close means best is probably very close
        rank += recognizer.config.rankings_second_is_close_add
    return int(np.clip(rank, 1, 10))


def _calc_rank(
    confidences,
    top_match_tups,
    rank_minus,
    offset_seconds,
    num_matches_tups: tuple = None,
    num_match: int = None,
):
    top_match = (confidences[0], offset_seconds[0])
    rank_delta = 0
    list_less_times = list(
        filter(
            lambda x: abs(x[1] - top_match[1]) > 0.15,
            zip(confidences, offset_seconds),
        )
    )
    if len(list_less_times) > 1:
        confidences, _ = list(zip(*list_less_times))
        third_match = confidences[1] if len(confidences) > 1 else 0

        second_match = confidences[0] if len(confidences) > 1 else 0
        top_three_list = np.array([top_match[0], second_match, third_match])
        top_three_list -= top_three_list[2]
        second_proportion = (
            top_three_list[1] / top_three_list[0] if top_three_list[1] > 0 else 0
        )
        if second_proportion > 0.75:
            rank_delta = 2
    else:
        third_match = 0
    top_match = top_match[0]
    rank_list = [top_match > x[0] for x in top_match_tups]
    rank = top_match_tups[rank_list.index(True)][1]
    third_match_list = [third_match >= top_match * x[0] for x in rank_minus]
    rank -= rank_minus[third_match_list.index(True)][1]
    if num_matches_tups is not None:
        num_matches_tup_list = [num_match <= x[0] for x in num_matches_tups]
        rank -= num_matches_tups[num_matches_tup_list.index(True)][1]
    return rank - rank_delta


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
