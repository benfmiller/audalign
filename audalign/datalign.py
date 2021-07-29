import numpy as np


def rank_alignment(alignment):
    ranks = {}
    if "fine_match_info" in alignment.keys():
        ranks["fine_match_info"] = _rank_alignment(alignment["fine_match_info"])
    ranks["match_info"] = _rank_alignment(alignment=alignment["match_info"])
    return ranks


def _rank_alignment(alignment):
    new_ranks = {}
    if alignment is None:
        return 0
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
    """Big decision tree

    Locality is tricky because the locality width affect confidence so much.
    The current values reflect an average locality value (20ish to 3 ish).
    Should still be taken with a grain of salt.
    """
    offset_seconds = alignment["offset_seconds"]
    if "confidence" in alignment.keys() and "scaling_factor" not in alignment.keys():
        # fingerprints
        rank_minus = (
            (0.95, 4),
            (0.9, 3),
            (0.85, 2),
            (0.8, 1),
            (0.7, 0),
            (0.6, -1),
            (0.45, -2),
            (0.0, 0),
        )
        confidences = alignment["confidence"]
        if alignment["locality_seconds"][0] is not None:  # locality
            top_match_tups = (
                (100, 10),
                (80, 9),
                (60, 8),
                (40, 7),
                (30, 6),
                (15, 5),
                (12, 4),
                (8, 3),
                (0, 1),
            )
        else:  # no locality
            top_match_tups = (
                (500, 10),
                (200, 9),
                (100, 8),
                (70, 7),
                (30, 6),
                (15, 5),
                (10, 4),
                (0, 1),
            )
        rank = _calc_rank(
            confidences=confidences,
            top_match_tups=top_match_tups,
            rank_minus=rank_minus,
            offset_seconds=offset_seconds,
        )
        if len(offset_seconds) > 1 and abs(offset_seconds[0] - offset_seconds[1]) < 0.5:
            # second best is very close means probably very close
            rank += 1
    elif "confidence" not in alignment.keys():
        # visual
        rank_minus = ((0.98, 4), (0.96, 3), (0.94, 2), (0.91, 1), (0.0, 0))
        top_match_tups = (
            (0.68, 10),
            (0.66, 9),
            (0.63, 8),
            (0.61, 7),
            (0.59, 6),
            (0.55, 5),
            (0.5, 4),
            (0, 1),
        )
        confidences = alignment["ssim"]
        top_num_match = alignment["num_matches"][0]
        num_matches_tups = (
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
        rank = _calc_rank(
            confidences=confidences,
            top_match_tups=top_match_tups,
            rank_minus=rank_minus,
            offset_seconds=offset_seconds,
            num_matches_tups=num_matches_tups,
            num_match=top_num_match,
        )
    elif "offset_frames" in alignment.keys():
        # correlation_spectrogram
        rank_minus = (
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
        confidences = alignment["confidence"]
        confidences = [x * alignment["scaling_factor"] for x in confidences]
        if alignment["locality_seconds"][0] is not None:  # there is locality
            top_match_tups = (
                (9, 10),
                (7.5, 9),
                (6, 8),
                (4, 7),
                (3, 6),
                (2, 5),
                (1, 4),
                (0, 1),
            )
        else:  # no locality
            top_match_tups = (
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
        rank = _calc_rank(
            confidences=confidences,
            top_match_tups=top_match_tups,
            rank_minus=rank_minus,
            offset_seconds=offset_seconds,
        )
        if len(offset_seconds) > 1 and abs(offset_seconds[0] - offset_seconds[1]) < 0.5:
            # second best is very close means probably very close
            rank += 1
    else:
        # Correlation
        rank_minus = (
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
        confidences = alignment["confidence"]
        confidences = [x * alignment["scaling_factor"] for x in confidences]
        if alignment["locality_seconds"][0] is not None:  # there is locality
            top_match_tups = (
                (15, 10),
                (12, 9),
                (9, 8),
                (7, 7),
                (5.5, 6),
                (3, 5),
                (2, 4),
                (0, 1),
            )
        else:  # no locality
            top_match_tups = (
                (8, 10),
                (6, 9),
                (4, 8),
                (3, 7),
                (2, 6),
                (1.5, 5),
                (1, 4),
                (0, 1),
            )
        rank = _calc_rank(
            confidences=confidences,
            top_match_tups=top_match_tups,
            rank_minus=rank_minus,
            offset_seconds=offset_seconds,
        )
        if len(offset_seconds) > 1 and abs(offset_seconds[0] - offset_seconds[1]) < 0.5:
            # second best is very close means probably very close
            rank += 1
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
