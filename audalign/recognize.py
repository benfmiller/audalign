import audalign.fingerprint as fingerprint
import time
import os


def recognize(
    audalign_object,
    file_path: str,
    filter_matches: int,
    locality: float,
    locality_filter_prop: float,
    start_end: tuple,
    max_lags: float = None,
):
    """
    Recognizes given file against already fingerprinted files

    Args
        file_path (str): file path of target file
        filter_matches (int): only returns information on match counts greater than filter_matches

    Returns
    -------
        match_result (dict): dictionary containing match time and match info

        or

        None : if no match
    """
    if filter_matches is None:
        filter_matches = 1
    if locality_filter_prop is None:
        locality_filter_prop = 0.6
    elif locality_filter_prop > 1.0:
        locality_filter_prop = 1.0
    if locality is not None:  # convert from seconds to samples
        locality = max(  # turns into frames
            int(
                locality
                // (
                    fingerprint.DEFAULT_WINDOW_SIZE
                    / fingerprint.DEFAULT_FS
                    * fingerprint.DEFAULT_OVERLAP_RATIO
                )
            ),
            1,
        )
    if max_lags is not None:
        max_lags = max(  # turns into frames
            int(
                max_lags
                // (
                    fingerprint.DEFAULT_WINDOW_SIZE
                    / fingerprint.DEFAULT_FS
                    * fingerprint.DEFAULT_OVERLAP_RATIO
                )
            ),
            1,
        )

    t = time.time()
    matches = find_matches(audalign_object, file_path, start_end=start_end)
    if locality:
        rough_match = locality_align_matches(matches, locality, locality_filter_prop)
    else:
        rough_match = align_matches(matches)

    filter_set = False

    if filter_matches != 1:
        filter_set = True

    file_match = None
    if len(rough_match) > 0:
        file_match = process_results(
            audalign_object,
            rough_match,
            locality,
            filter_matches,
            filter_set,
            max_lags=max_lags,
        )
    t = time.time() - t

    result = {}

    if file_match:
        result["match_time"] = t
        result["match_info"] = file_match
        return result

    return None


def find_matches(audalign_object, file_path, start_end):
    """
    fingerprints target file, then finds every occurence of exact same hashes in already
    fingerprinted files

    Args
        samples (array of decoded file): array of decoded file from filehandler.read
        file_name (str): base name of target file

    Returns
    -------
        Matches(list[str, int, int, int]): list of all matches, file_name match, corresponding offset, target location, file_match offset
    """
    file_name = os.path.basename(file_path)

    target_mapper = {}

    if file_name not in audalign_object.file_names:
        fingerprints = audalign_object._fingerprint_file(file_path, start_end=start_end)
        target_mapper = fingerprints[1]
    else:
        for audio_file in audalign_object.fingerprinted_files:
            if audio_file[0] == file_name:
                target_mapper = audio_file[1]
                break

    matches = []

    print(f"{file_name}: Finding Matches...  ", end="")
    for audio_file in audalign_object.fingerprinted_files:
        if audio_file[0].lower() != file_name.lower():
            already_hashes = audio_file[1]
            for t_hash in target_mapper.keys():
                if t_hash in already_hashes.keys():
                    for t_offset in target_mapper[t_hash]:
                        for a_offset in already_hashes[t_hash]:
                            sample_difference = a_offset - t_offset
                            matches.append(
                                [audio_file[0], sample_difference, t_offset, a_offset]
                            )
    return matches


def align_matches(matches: list):
    """
    takes matches from find_matches and converts it to a dictionary of counts per offset and file name

    Args
        matches (list[str, int]): list of matches from find_matches

    Returns
    -------
        sample_difference_counter (dict{str{int}}): of the form dict{file_name{number of matching offsets}}
    """

    print("Aligning matches")
    sample_difference_counter = {}
    for file_name, sample_difference, _, _ in matches:
        if file_name not in sample_difference_counter:
            sample_difference_counter[file_name] = {}
        if sample_difference not in sample_difference_counter[file_name]:
            sample_difference_counter[file_name][sample_difference] = [0, None]
        sample_difference_counter[file_name][sample_difference][0] += 1

    return sample_difference_counter


def locality_align_matches(matches: list, locality: int, locality_filter_prop: int):

    print("Aligning matches")
    sample_difference_counter = {}
    file_dict = {}

    # converting matches into file_dict of matches
    for file_name, sample_difference, t_offset, a_offset in matches:
        if file_dict.get(file_name) is None:
            file_dict[file_name] = []
        file_dict[file_name].append((sample_difference, t_offset, a_offset))

    # shifting windows for each filename match
    for name in file_dict.keys():
        temp_file_dict = {}
        start_window = 0
        end_window = 1
        last_end = 1

        # sorts by t_offset
        file_dict[name] = sorted(file_dict[name], key=lambda x: x[1])

        while (
            end_window < len(file_dict[name]) - 1
            and file_dict[name][end_window][1] - file_dict[name][start_window][1]
            <= locality
        ):
            end_window += 1
            last_end = end_window

        # moves end while there's room and locality is
        while True:  # end_window <= len(file_dict[name]):

            # {(toff, aoff): {samp_diff : confidence}}
            toff_dict = find_loc_matches(
                file_dict[name][start_window:end_window], locality
            )

            # combines and turns into {offset: [confidence, [loc_tups]]}
            for tup, samp_dict in toff_dict.items():
                for samp_diff, confidence in samp_dict.items():
                    if temp_file_dict.get(samp_diff) is None:
                        temp_file_dict[samp_diff] = [confidence, []]
                    elif temp_file_dict[samp_diff][0] < confidence:
                        temp_file_dict[samp_diff][0] = confidence
                    temp_file_dict[samp_diff][1] += [(*tup, confidence)]

            # breaks out of while if at end of file and within locality
            if end_window >= len(file_dict[name]):
                break

            while True:
                start_window += 1
                while (
                    end_window <= len(file_dict[name]) - 1
                    and file_dict[name][end_window][1]
                    - file_dict[name][start_window][1]
                    <= locality
                ):
                    end_window += 1
                if end_window >= len(file_dict[name]):
                    break
                if end_window > last_end:
                    last_end = end_window
                    break

        # # filter to top 30
        if len(temp_file_dict.keys()) > 30:
            temp_file_list = [
                (samp_diff, conf_loc) for samp_diff, conf_loc in temp_file_dict.items()
            ]
            temp_file_list = sorted(
                temp_file_list, key=lambda x: x[1][0], reverse=True
            )  # sort by confidence
            temp_file_dict = {}
            for i in range(30):
                temp_file_dict[temp_file_list[i][0]] = temp_file_list[i][1]

        # locality_filter_prop
        for _, matches in temp_file_dict.items():
            index = 0
            while index < len(matches[1]):
                if matches[1][index][2] < matches[0] * locality_filter_prop:
                    matches[1].pop(index)
                    continue
                index += 1

        if len(temp_file_dict) > 0:
            sample_difference_counter[name] = temp_file_dict

    # return {filename: {offset: [confidence, [loc_tups]]}}
    return sample_difference_counter


def find_loc_matches(matches_list: list, locality: int):
    """receives from align matches locality,
        matcheslist = [(sample_difference, t_offset, a_offset)]

    Args:
        matches_list (list): [(sample_difference, t_offset, a_offset)]
        locality (int): [description]

    Returns:
        [dict]: {(toff, aoff): {samp_diff : confidence}}
    """
    # matches_list = list(set(matches_list))

    a_matches = sorted(matches_list, key=lambda x: x[2])
    temp_file_dict = {}
    start_window = 0
    end_window = 0
    last_end = 0

    while (
        end_window < len(a_matches) - 1
        and a_matches[end_window + 1][2] - a_matches[start_window][2] <= locality
    ):
        end_window += 1
        last_end = end_window

    while True:  # end_window <= len(a_matches):

        loc_tup = (
            ((matches_list[-1][1] - matches_list[0][1]) // 2) + matches_list[0][1],
            ((a_matches[end_window][2] - a_matches[start_window][2]) // 2)
            + a_matches[start_window][2],
        )
        # loc_tup = ( # Old version
        #     (matches_list[-1][1] - matches_list[0][1]) // 2,
        #     (a_matches[end_window][2] - a_matches[start_window][2]) // 2,
        # )
        temp_file_dict[loc_tup] = {}
        for sample_difference, t_offset, a_offset in a_matches[start_window:end_window]:
            if sample_difference not in temp_file_dict[loc_tup].keys():
                temp_file_dict[loc_tup][sample_difference] = 0
            temp_file_dict[loc_tup][sample_difference] += 1
        # gives us temp_file_dict--- {(toff, aoff): {samp_diff : confidence}}

        # breaks out of while if at end of file and within locality

        if end_window >= len(a_matches) - 1:
            break

        while True:
            start_window += 1
            while (
                end_window < len(a_matches) - 1
                and a_matches[end_window + 1][2] - a_matches[start_window][2]
                <= locality
            ):
                end_window += 1
            if end_window >= len(a_matches) - 1:
                break
            if end_window > last_end:
                last_end = end_window
                break

    return temp_file_dict


def process_results(
    audalign_object,
    results,
    locality,
    filter_matches: int = 1,
    filter_set: bool = False,
    max_lags: float = None,
):
    """
    Takes matches from align_matches, filters and orders them, returns dictionary of match info

    Args
        results (dict{str{int}}): of the form dict{file_name{number of matching offsets}}
        filter_matches (int): cutout all matches equal to or less than in frequency, goes down if no matches found above filter
        filter_set (bool): if the filter is manually set, doesn't lower filter if no results

    Returns
    -------
        match_info (dict{dict{}}): dict of file_names with match info as values
    """

    complete_match_info = {}

    for file_name in results.keys():
        match_offsets = []
        offset_count = []
        offset_diff = []
        offset_loc = []
        for sample_difference, num_of_matches_loc in results[file_name].items():
            match_offsets.append((num_of_matches_loc, sample_difference))
        match_offsets = sorted(match_offsets, reverse=True, key=lambda x: x[0][0])
        if match_offsets[0][0][0] <= filter_matches:
            continue
        if max_lags is not None:
            i = 0
            while i < len(match_offsets):
                if abs(match_offsets[i][1]) > max_lags:
                    match_offsets.pop(i)
                    continue
                i += 1
        for i in match_offsets:
            if i[0][0] <= filter_matches:
                continue
            offset_count.append(i[0][0])
            offset_loc.append(i[0][1])
            offset_diff.append(i[1])

        complete_match_info[file_name] = {}
        complete_match_info[file_name][audalign_object.CONFIDENCE] = offset_count
        complete_match_info[file_name][audalign_object.OFFSET_SAMPLES] = offset_diff
        complete_match_info[file_name][audalign_object.LOCALITY] = offset_loc
        if locality:
            complete_match_info[file_name][
                audalign_object.LOCALITY + "_setting"
            ] = round(
                float(locality)
                / fingerprint.DEFAULT_FS
                * fingerprint.DEFAULT_WINDOW_SIZE
                * fingerprint.DEFAULT_OVERLAP_RATIO,
                5,
            )

        else:
            complete_match_info[file_name][audalign_object.LOCALITY + "_setting"] = None

        # calculate seconds
        complete_match_info[file_name][audalign_object.OFFSET_SECS] = []
        for i in offset_diff:
            nseconds = round(
                float(i)
                / fingerprint.DEFAULT_FS
                * fingerprint.DEFAULT_WINDOW_SIZE
                * fingerprint.DEFAULT_OVERLAP_RATIO,
                5,
            )
            complete_match_info[file_name][audalign_object.OFFSET_SECS].append(nseconds)

        # Calculate locality tuples seconds
        new_offset_loc = []
        complete_match_info[file_name][audalign_object.LOCALITY_SECS] = []
        for instance in range(len(offset_loc)):
            if locality:
                new_offset_loc += [[]]
                for location in range(len(offset_loc[instance])):
                    new_offset_loc[instance] += [
                        (
                            round(
                                float(offset_loc[instance][location][0])
                                / fingerprint.DEFAULT_FS
                                * fingerprint.DEFAULT_WINDOW_SIZE
                                * fingerprint.DEFAULT_OVERLAP_RATIO,
                                5,
                            ),
                            round(
                                float(offset_loc[instance][location][1])
                                / fingerprint.DEFAULT_FS
                                * fingerprint.DEFAULT_WINDOW_SIZE
                                * fingerprint.DEFAULT_OVERLAP_RATIO,
                                5,
                            ),
                            offset_loc[instance][location][2],
                        )
                    ]
            else:
                new_offset_loc += [None]
            complete_match_info[file_name][audalign_object.LOCALITY_SECS].append(
                new_offset_loc[instance]
            )

    if len(complete_match_info) == 0 and filter_set == False:
        return process_results(
            audalign_object,
            results,
            locality,
            filter_matches=filter_matches - 1,
        )

    return complete_match_info
