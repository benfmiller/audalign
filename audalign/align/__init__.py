import multiprocessing
import os
import typing
from functools import partial
from posixpath import basename

import audalign
import audalign.filehandler as filehandler
import tqdm
from audalign.recognizers import BaseRecognizer


def _align(
    recognizer: BaseRecognizer,
    filename_list: typing.Union[str, list],
    file_dir: typing.Optional[str],
    destination_path: str = None,
    write_extension: str = None,
    write_multi_channel: bool = False,
    fine_aud_file_dict: dict = None,
    target_aligning: bool = False,
):
    if destination_path is not None and not os.path.exists(destination_path):
        raise ValueError(f'destination_path "{destination_path}" does not exist')

    file_names_to_align = recognizer.align_get_file_names(
        file_list=filename_list,
        file_dir=file_dir,
        target_aligning=target_aligning,
        fine_aud_file_dict=fine_aud_file_dict,
    )

    file_list, dir_or_list = set_list_and_dir(
        filename_list=filename_list,
        file_dir=file_dir,
        target_aligning=target_aligning,
        fine_aud_file_dict=fine_aud_file_dict,
    )

    total_alignment, file_names_and_paths = calc_alignments(
        recognizer=recognizer,
        file_names_to_align=file_names_to_align,
        file_list=file_list,
        dir_or_list=dir_or_list,
        target_aligning=target_aligning,
        fine_aud_file_dict=fine_aud_file_dict,
    )

    files_shifts = calc_final_alignments(
        recognizer=recognizer,
        filename_list=filename_list,
        file_dir=file_dir,
        total_alignment=total_alignment,
        destination_path=destination_path,
        file_names_and_paths=file_names_and_paths,
        write_extension=write_extension,
        write_multi_channel=write_multi_channel,
        target_aligning=target_aligning,
        fine_aligning=fine_aud_file_dict is not None,
    )

    recognizer.align_post_hook(
        file_list=file_list,
        dir_or_list=dir_or_list,
        target_aligning=target_aligning,
        fine_aud_file_dict=fine_aud_file_dict,
    )

    if not files_shifts:
        print(f"0 out of {len(file_names_and_paths)} found and aligned")
        return

    print(f"{len(files_shifts)} out of {len(file_names_and_paths)} found and aligned")

    files_shifts["match_info"] = total_alignment
    files_shifts["names_and_paths"] = file_names_and_paths

    recognizer.align_stat_print()

    return files_shifts


def set_list_and_dir(
    filename_list,
    file_dir,
    target_aligning: bool,
    fine_aud_file_dict: typing.Optional[dict],
):
    if target_aligning:  # For target aligning
        file_list = zip(filename_list, ["_"] * len(filename_list))
        dir_or_list = file_dir
    elif file_dir:  # For regular aligning
        file_list = audalign.filehandler.find_files(file_dir)
        dir_or_list = file_dir
    elif fine_aud_file_dict is not None:  # For fine_aligning
        file_list = zip(fine_aud_file_dict.keys(), ["_"] * len(fine_aud_file_dict))
        dir_or_list = list(fine_aud_file_dict.keys())
    else:  # For align_files
        file_list = zip(filename_list, ["_"] * len(filename_list))
        dir_or_list = filename_list
    return file_list, dir_or_list


def calc_alignments(
    recognizer: BaseRecognizer,
    file_names_to_align: list,
    file_list,
    dir_or_list,
    target_aligning: bool,
    fine_aud_file_dict: typing.Optional[dict],
):
    total_alignment = {}
    file_names_and_paths = {}
    # Get matches and paths

    if recognizer.check_align_hook(
        file_list=file_list,
        dir_or_list=dir_or_list,
        target_aligning=target_aligning,
        fine_aud_file_dict=fine_aud_file_dict,
    ):

        _calc_alignments = recognizer.align_hook(
            file_list=file_list,
            dir_or_list=dir_or_list,
            target_aligning=target_aligning,
            fine_aud_file_dict=fine_aud_file_dict,
        )
        temp_file_list = []
        for file_path, _ in file_list:
            if os.path.basename(file_path) in file_names_to_align:
                temp_file_list += [file_path]

        with multiprocessing.Pool(recognizer.config.num_processors) as pool:
            results_list = pool.map(_calc_alignments, tqdm.tqdm(list(temp_file_list)))
            pool.close()
            pool.join()

        for i in results_list:
            if i is not None:
                file_names_and_paths[os.path.basename(i["filename"])] = i["filename"]
                total_alignment[os.path.basename(i["filename"])] = i
                i.pop("filename")

    else:
        for file_path, _ in file_list:
            name = os.path.basename(file_path)
            if name in file_names_to_align:
                alignment = recognizer._align(file_path, dir_or_list)
                file_names_and_paths[name] = file_path
                total_alignment[name] = alignment
    return total_alignment, file_names_and_paths


def calc_final_alignments(
    recognizer: BaseRecognizer,
    filename_list,
    file_dir,
    total_alignment,
    destination_path,
    file_names_and_paths,
    write_extension,
    write_multi_channel: bool,
    target_aligning,
    fine_aligning: bool,
):
    # TODO: refactor into a recursive/graph alignment finding method
    # Would allow for overlaps with overlaps with overlaps
    # Without only relying on most matched file.
    # Use filter to clear out excess matches
    files_shifts = find_most_matches(
        total_alignment, strength_stat=recognizer.config.CONFIDENCE
    )
    if not files_shifts:
        return
    files_shifts = find_matches_not_in_file_shifts(
        total_alignment, files_shifts, strength_stat=recognizer.config.CONFIDENCE
    )

    if target_aligning:
        for file_path, _ in audalign.filehandler.find_files(file_dir):
            if (
                os.path.basename(file_path)
                in total_alignment[os.path.basename(filename_list[0])][
                    "match_info"
                ].keys()
            ):
                file_names_and_paths[os.path.basename(file_path)] = file_path
    if fine_aligning is False:
        max_shift = max(files_shifts.values())
        for name in files_shifts.keys():
            files_shifts[name] = max_shift - files_shifts[name]
    else:
        min_shift = min(files_shifts.values())
        for name in files_shifts.keys():
            files_shifts[name] = files_shifts[name] - min_shift

    if destination_path:
        try:
            filehandler.shift_write_files(
                files_shifts,
                destination_path,
                file_names_and_paths,
                write_extension,
                write_multi_channel,
            )
        except PermissionError:
            print("Permission Denied for write align")
    return files_shifts


def find_most_matches(
    total_alignment, strength_stat: str = "confidence", match_index: int = 0
):
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
            running_strength += match[strength_stat][match_index]
        if running_strength > total_match_strength:
            total_match_strength = running_strength
            most_matches_file["most_matches"] = file_match
            most_matches_file["match_info"] = total_alignment[file_match]["match_info"]

    files_shifts = {}
    files_shifts[most_matches_file["most_matches"]] = 0

    for name, file_match in most_matches_file["match_info"].items():
        files_shifts[name] = file_match[audalign.BaseConfig.OFFSET_SECS][match_index]

    return files_shifts


def find_matches_not_in_file_shifts(
    total_alignment,
    files_shifts,
    strength_stat: str = "confidence",
    match_index: str = 0,
):
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
                        nmatch_wt_most[main_name][
                            audalign.BaseConfig.OFFSET_SECS
                        ] = None
                    if (
                        file_match[strength_stat][match_index]
                        > nmatch_wt_most[main_name]["match_strength"]
                    ):
                        nmatch_wt_most["match_strength"] = file_match[strength_stat][
                            match_index
                        ]
                        nmatch_wt_most[audalign.BaseConfig.OFFSET_SECS] = (
                            file_match[audalign.BaseConfig.OFFSET_SECS][match_index]
                            - files_shifts[match_name]
                        )

    return files_shifts


def combine_fine(results: dict, new_results: dict):
    fine_match_info = new_results.pop("match_info")
    temp_names_and_paths = new_results.pop("names_and_paths")
    max_shift = max(new_results.values())
    for name in new_results.keys():
        new_results[name] = max_shift - new_results[name] + results[name]
    min_shift = min(new_results.values())
    for name in new_results.keys():
        new_results[name] = new_results[name] - min_shift
    new_results["fine_match_info"] = fine_match_info
    new_results["match_info"] = results["match_info"]
    if results.get("rankings") is not None:
        new_results["rankings"] = results["rankings"]
    new_results["names_and_paths"] = temp_names_and_paths
    return new_results


def recalc_shifts_index(
    results: dict,
    key=None,
    match_index: int = 0,
    fine_match_index: int = 0,
    strength_stat: str = None,
    fine_strength_stat=None,
) -> dict:
    if key is None:
        if "fine_match_info" in results:
            key = "fine_match_info"
        else:
            key = "match_info"
    if key not in ["match_info", "fine_match_info", "only_fine_match_info"]:
        raise ValueError(
            f'key must be "match_info", "fine_match_info", or "only_fine_match_info", not "{key}"'
        )
    temp_info_key = "match_info" if key != "only_fine_match_info" else "fine_match_info"

    if strength_stat is None:
        strength_stat = "confidence"
        _ = results[temp_info_key]
        _ = _[list(_.keys())[0]]
        _ = _["match_info"]
        _ = _[list(_.keys())[0]]
        if "confidence" not in _.keys():
            strength_stat = "ssim"

    files_shifts = _calc_shifts_index(
        results[temp_info_key], strength_stat, match_index=match_index
    )
    if temp_info_key == "match_info":
        max_shift = max(files_shifts.values())
        for name in files_shifts.keys():
            files_shifts[name] = max_shift - files_shifts[name]
    else:
        min_shift = min(files_shifts.values())
        for name in files_shifts.keys():
            files_shifts[name] = files_shifts[name] - min_shift
    for name in files_shifts.keys():
        results[name] = files_shifts[name]

    if key == "fine_match_info":
        if fine_strength_stat is None:
            fine_strength_stat = "confidence"
            _ = results["fine_match_info"]
            _ = _[list(_.keys())[0]]
            _ = _["match_info"]
            _ = _[list(_.keys())[0]]
            if "confidence" not in _.keys():
                fine_strength_stat = "ssim"
        files_shifts = _calc_shifts_index(
            results["fine_match_info"], fine_strength_stat, match_index=fine_match_index
        )
        max_shift = max(files_shifts.values())
        for name in files_shifts.keys():
            files_shifts[name] = max_shift - files_shifts[name] + results[name]
        min_shift = min(files_shifts.values())
        for name in files_shifts.keys():
            results[name] = files_shifts[name] - min_shift

    for name in results.keys():
        if name not in [
            "match_info",
            "fine_match_info",
            "rankings",
            "names_and_paths",
            *list(files_shifts.keys()),
        ]:
            results.pop(name)

    return results


def _calc_shifts_index(info_to_use, strength_stat, match_index):
    files_shifts = find_most_matches(
        info_to_use, strength_stat=strength_stat, match_index=match_index
    )
    if not files_shifts:
        return
    return find_matches_not_in_file_shifts(
        info_to_use,
        files_shifts,
        strength_stat=strength_stat,
        match_index=match_index,
    )
