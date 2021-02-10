import audalign


def find_most_matches(total_alignment, strength_stat: str = "confidence"):
    """
    Finds the file that matches with the most files and has the most matches, returns its matches and shifts

    Args
        total_alignment (dict{dict{}}): dict of recognize results

    Returns
    -------
        files_shifts (dict{float}): dict with file names as keys and shift amounts as values
    """

    most_matches = 0
    most_matches_file = {}
    most_matches_file["tied"] = []
    most_matches_file["most_matches"] = None

    no_matches_list = []

    # find file with most matches
    for name, match in total_alignment.items():
        if match:
            if (len(match["match_info"])) > most_matches:
                most_matches = len(match["match_info"])
                most_matches_file["most_matches"] = name
                most_matches_file["tied"] = [name]
            elif (len(match["match_info"])) == most_matches:
                most_matches_file["tied"] += [name]
        else:
            no_matches_list += [name]

    if len(no_matches_list) == len(total_alignment):
        print("No matches detected")
        return

    total_match_strength = 0  # total match count of strongest match per file

    # Get match info for file with strongest matches
    for file_match in most_matches_file["tied"]:
        running_strength = 0
        for _, match in total_alignment[file_match]["match_info"].items():
            running_strength += match[strength_stat][0]
        if running_strength > total_match_strength:
            total_match_strength = running_strength
            most_matches_file["most_matches"] = file_match
            most_matches_file["match_info"] = total_alignment[file_match]["match_info"]

    files_shifts = {}
    files_shifts[most_matches_file["most_matches"]] = 0

    for name, file_match in most_matches_file["match_info"].items():
        files_shifts[name] = file_match[audalign.Audalign.OFFSET_SECS][0]

    return files_shifts


def find_matches_not_in_file_shifts(total_alignment, files_shifts):
    """
    Checks to find files that match with files that match with most matched file and update files_shifts

    Args
        total_alignment (dict{dict{}}): dict of recognize results
        files_shifts (dict{float}): dict with file names as keys and shift amounts as values

    Returns
    -------
        files_shifts (dict{float}): dict with file names as keys and shift amounts as values (min of zero now)
    """

    nmatch_wt_most = {}

    # Finds files that aren't in files_shifts that match with files in files_shifts
    for main_name, file_matches in total_alignment.items():
        if file_matches and main_name not in files_shifts.keys():
            for match_name, file_match in file_matches["match_info"].items():
                if match_name in files_shifts:
                    if main_name not in nmatch_wt_most:
                        nmatch_wt_most[main_name] = {}
                        nmatch_wt_most[main_name]["match_strength"] = 0
                        nmatch_wt_most[main_name][audalign.Audalign.OFFSET_SECS] = None
                    if (
                        file_match[audalign.Audalign.CONFIDENCE][0]
                        > nmatch_wt_most[main_name]["match_strength"]
                    ):
                        nmatch_wt_most["match_strength"] = file_match[
                            audalign.Audalign.CONFIDENCE
                        ][0]
                        nmatch_wt_most[audalign.Audalign.OFFSET_SECS] = (
                            file_match[audalign.Audalign.OFFSET_SECS][0]
                            - files_shifts[match_name]
                        )

    return files_shifts
