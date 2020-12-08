import audalign.fingerprint as fingerprint
import time
import os


def recognize(audalign_object, file_path, filter_matches, locality):
    """
    Recognizes given file against already fingerprinted files

    Parameters
    ----------
    file_path : str
        file path of target file
    filter_matches : int
        only returns information on match counts greater than filter_matches

    Returns
    -------
    match_result : dict
        dictionary containing match time and match info

        or

        None : if no match
    """

    t = time.time()
    matches = find_matches(audalign_object, file_path)
    rough_match = align_matches(matches, locality)

    filter_set = False

    if filter_matches != 1:
        filter_set = True

    file_match = None
    if len(rough_match[0]) > 0:
        file_match = process_results(
            audalign_object, rough_match, locality, filter_matches, filter_set
        )
    t = time.time() - t

    result = {}

    if file_match:
        result["match_time"] = t
        result["match_info"] = file_match
        return result

    return None


def find_matches(audalign_object, file_path):
    """
    fingerprints target file, then finds every occurence of exact same hashes in already
    fingerprinted files

    Parameters
    ----------
    samples : array of decoded file
        array of decoded file from filehandler.read
    file_name : str
        base name of target file

    Returns
    -------
    Matches: list[str, int, int, int]
        list of all matches, file_name match, corresponding offset, target location, file_match offset
    """
    file_name = os.path.basename(file_path)

    target_mapper = {}

    if file_name not in audalign_object.file_names:
        fingerprints = audalign_object._fingerprint_file(file_path)
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


def align_matches(matches, locality):
    """
    takes matches from find_matches and converts it to a dictionary of counts per offset and file name

    Parameters
    ----------
    matches : list[str, int]
        list of matches from find_matches

    Returns
    -------
    sample_difference_counter : dict{str{int}}
        of the form dict{file_name{number of matching offsets}}
    """

    print("Aligning matches")

    if locality:
        sample_difference_counter = {}
        for file_name, sample_difference, _, _ in matches:
            if file_name not in sample_difference_counter:
                sample_difference_counter[file_name] = {}
            if sample_difference not in sample_difference_counter[file_name]:
                sample_difference_counter[file_name][sample_difference] = 0
            sample_difference_counter[file_name][sample_difference] += 1

        return (sample_difference_counter, None)
    else:
        sample_difference_counter = {}
        for file_name, sample_difference, t_offset, a_offset in matches:
            if file_name not in sample_difference_counter:
                sample_difference_counter[file_name] = {}
            if sample_difference not in sample_difference_counter[file_name]:
                sample_difference_counter[file_name][sample_difference] = 0
            sample_difference_counter[file_name][sample_difference] += 1

        return (sample_difference_counter, None)


def process_results(
    audalign_object, results, locality, filter_matches=1, filter_set=False
):
    """
    Takes matches from align_matches, filters and orders them, returns dictionary of match info

    Parameters
    ----------
    results : dict{str{int}}
        of the form dict{file_name{number of matching offsets}}

    filter_matches : int
        cutout all matches equal to or less than in frequency, goes down if no matches found above filter

    filter_set : bool
        if the filter is manually set, doesn't lower filter if no results

    Returns
    -------
    match_info : dict{dict{}}
        dict of file_names with match info as values
    """

    results = results[0]
    complete_match_info = {}

    for file_name in results.keys():
        match_offsets = []
        offset_count = []
        offset_diff = []
        for sample_difference, num_of_matches in results[file_name].items():
            match_offsets.append((num_of_matches, sample_difference))
        match_offsets = sorted(match_offsets, reverse=True, key=lambda x: x[0])
        if match_offsets[0][0] <= filter_matches:
            continue
        for i in match_offsets:
            if i[0] <= filter_matches:
                continue
            offset_count.append(i[0])
            offset_diff.append(i[1])

        complete_match_info[file_name] = {}
        complete_match_info[file_name][audalign_object.CONFIDENCE] = offset_count
        complete_match_info[file_name][audalign_object.OFFSET_SAMPLES] = offset_diff

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

    if len(complete_match_info) == 0 and filter_set == False:
        return process_results(
            audalign_object,
            (results, None),
            locality,
            filter_matches=filter_matches - 1,
        )

    return complete_match_info
