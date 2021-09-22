from posixpath import basename
import audalign
import audalign.filehandler as filehandler
import multiprocessing
import os
from functools import partial
import tqdm
import warnings


def _align(
    ada_obj,
    filename_list: str,
    file_dir: str,
    destination_path: str = None,
    write_extension: str = None,
    technique: str = "fingerprints",
    filter_matches: float = None,
    locality: float = None,
    locality_filter_prop: float = None,
    cor_sample_rate: int = None,
    max_lags: float = None,
    fine_aud_file_dict: dict = None,
    target_aligning: bool = False,
    target_start_end: tuple = None,
    alternate_strength_stat: str = None,
    volume_threshold: float = 216,
    volume_floor: float = 10.0,
    vert_scaling: float = 1.0,
    horiz_scaling: float = 1.0,
    img_width: float = 1.0,
    calc_mse: bool = False,
    load_fingerprints: str = None,
    **kwargs,
):
    if destination_path is not None and not os.path.exists(destination_path):
        raise ValueError(f'destination_path "{destination_path}" does not exist')
    ada_obj.file_names, temp_file_names = [], ada_obj.file_names
    ada_obj.fingerprinted_files, temp_fingerprinted_files = (
        [],
        ada_obj.fingerprinted_files,
    )
    ada_obj.total_fingerprints, temp_total_fingerprints = 0, ada_obj.total_fingerprints
    try:
        ada_obj.calc_mse = calc_mse
        ada_obj.alternate_strength_stat = alternate_strength_stat

        set_ada_file_names(
            ada_obj=ada_obj,
            filename_list=filename_list,
            file_dir=file_dir,
            destination_path=destination_path,
            technique=technique,
            target_aligning=target_aligning,
            target_start_end=target_start_end,
            fine_aud_file_dict=fine_aud_file_dict,
            load_fingerprints=load_fingerprints,
        )

        file_list, dir_or_list = set_list_and_dir(
            filename_list=filename_list,
            file_dir=file_dir,
            target_aligning=target_aligning,
            fine_aud_file_dict=fine_aud_file_dict,
        )

        total_alignment, file_names_and_paths = calc_alignments(
            ada_obj=ada_obj,
            file_list=file_list,
            dir_or_list=dir_or_list,
            technique=technique,
            filter_matches=filter_matches,
            locality=locality,
            locality_filter_prop=locality_filter_prop,
            max_lags=max_lags,
            target_start_end=target_start_end,
            target_aligning=target_aligning,
            cor_sample_rate=cor_sample_rate,
            volume_threshold=volume_threshold,
            volume_floor=volume_floor,
            vert_scaling=vert_scaling,
            horiz_scaling=horiz_scaling,
            img_width=img_width,
            fine_aud_file_dict=fine_aud_file_dict,
            **kwargs,
        )

        files_shifts = calc_final_alignments(
            ada_obj=ada_obj,
            filename_list=filename_list,
            file_dir=file_dir,
            technique=technique,
            total_alignment=total_alignment,
            destination_path=destination_path,
            file_names_and_paths=file_names_and_paths,
            write_extension=write_extension,
            target_aligning=target_aligning,
            fine_aligning=fine_aud_file_dict is not None,
        )

        if not files_shifts:
            return

        print(
            f"{len(files_shifts)} out of {len(file_names_and_paths)} found and aligned"
        )

        del ada_obj.calc_mse
        del ada_obj.alternate_strength_stat

        files_shifts["match_info"] = total_alignment
        files_shifts["names_and_paths"] = file_names_and_paths
        if technique == "fingerprints":
            print()
            print(f"Total fingerprints: {ada_obj.total_fingerprints}")

        return files_shifts

    finally:
        ada_obj.file_names = temp_file_names
        ada_obj.fingerprinted_files = temp_fingerprinted_files
        ada_obj.total_fingerprints = temp_total_fingerprints


def prelim_fingerprint_checks(ada_obj, target_file, directory_path):
    all_against_files = audalign.filehandler.find_files(directory_path)
    all_against_files_full = [x[0] for x in all_against_files]
    all_against_files_base = [os.path.basename(x) for x in all_against_files_full]
    if (
        os.path.basename(target_file) in all_against_files_base
        and target_file not in all_against_files_full
    ):
        for i, x in enumerate(all_against_files_full):
            if os.path.basename(target_file) == os.path.basename(x):
                all_against_files_full.pop(i)
                break
        all_against_files_full.append(target_file)
    elif os.path.basename(target_file) not in all_against_files_base:
        all_against_files_full.append(target_file)
    return all_against_files_full


def set_ada_file_names(
    ada_obj,
    filename_list,
    file_dir,
    destination_path,
    technique,
    target_aligning,
    target_start_end,
    fine_aud_file_dict,
    load_fingerprints,
):
    # Make target directory
    if destination_path:
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)

    if technique == "fingerprints":
        if load_fingerprints is not None:
            if os.path.exists(load_fingerprints):
                ada_obj.load_fingerprinted_files(load_fingerprints)
            else:
                warnings.warn(
                    f"load fingerprints given and fingerprinting, but load_fingerprints {load_fingerprints} not found"
                )
        if target_aligning:
            if target_start_end is not None:
                ada_obj.fingerprint_file(filename_list[0], start_end=target_start_end)
            file_dir = prelim_fingerprint_checks(
                ada_obj=ada_obj,
                target_file=filename_list[0],
                directory_path=file_dir,
            )
        if file_dir:
            ada_obj.fingerprint_directory(file_dir)
        else:
            ada_obj.fingerprint_directory(
                filename_list,
                _file_audsegs=fine_aud_file_dict,
            )
        if load_fingerprints is not None:
            if type(file_dir) == str:
                file_names = [
                    x for x in filehandler.get_audio_files_directory(file_dir)
                ]
            elif type(file_dir) == list:
                file_names = [os.path.basename(x) for x in file_dir]
            elif file_dir is None and fine_aud_file_dict is not None:
                file_names = [os.path.basename(x) for x in fine_aud_file_dict.keys()]
            elif filename_list is not None:
                file_names = [os.path.basename(x) for x in filename_list]

            ada_obj.file_names, temp_file_names = [], ada_obj.file_names
            ada_obj.fingerprinted_files, temp_fingerprinted_files = (
                [],
                ada_obj.fingerprinted_files,
            )
            ada_obj.total_fingerprints, temp_total_fingerprints = (
                0,
                ada_obj.total_fingerprints,
            )
            for loaded_file in temp_fingerprinted_files:
                if loaded_file[0] != None and loaded_file[0] in file_names:
                    ada_obj.fingerprinted_files.append(loaded_file)
                    ada_obj.file_names.append(loaded_file[0])
                    ada_obj.total_fingerprints += len(loaded_file[1])
    elif technique in ["correlation", "visual", "correlation_spectrogram"]:
        if target_aligning:
            ada_obj.file_names = [os.path.basename(x) for x in filename_list]
        elif file_dir:
            ada_obj.file_names = audalign.filehandler.get_audio_files_directory(
                file_dir
            )
        elif fine_aud_file_dict:
            ada_obj.file_names = [
                os.path.basename(x) for x in fine_aud_file_dict.keys()
            ]
        else:
            ada_obj.file_names = [os.path.basename(x) for x in filename_list]

        if technique == "visual":
            if ada_obj.alternate_strength_stat == "mse":
                ada_obj.calc_mse = True
            if ada_obj.alternate_strength_stat == "confidence":
                ada_obj.alternate_strength_stat = "ssim"

    else:
        raise ValueError(
            f'Technique parameter must be fingerprints, visual, correlation_spectrogram, or correlation, not "{technique}"'
        )


def set_list_and_dir(
    filename_list,
    file_dir,
    target_aligning,
    fine_aud_file_dict,
):
    if target_aligning:  # For target aligning
        file_list = zip(filename_list, ["_"] * len(filename_list))
        dir_or_list = file_dir
    elif file_dir:  # For regular aligning
        file_list = audalign.filehandler.find_files(file_dir)
        dir_or_list = file_dir
    elif fine_aud_file_dict:  # For fine_aligning
        file_list = zip(fine_aud_file_dict.keys(), ["_"] * len(fine_aud_file_dict))
        dir_or_list = list(fine_aud_file_dict.keys())
    else:  # For align_files
        file_list = zip(filename_list, ["_"] * len(filename_list))
        dir_or_list = filename_list
    return file_list, dir_or_list


def calc_alignments(
    ada_obj,
    file_list,
    dir_or_list,
    technique,
    filter_matches,
    locality,
    locality_filter_prop,
    max_lags,
    target_start_end,
    target_aligning,
    cor_sample_rate,
    volume_threshold,
    volume_floor,
    vert_scaling,
    horiz_scaling,
    img_width,
    fine_aud_file_dict,
    **kwargs,
):
    total_alignment = {}
    file_names_and_paths = {}
    # Get matches and paths

    if (
        ada_obj.multiprocessing == True
        and technique != "fingerprints"
        and not target_aligning
    ):

        if technique == "visual":
            _calc_alignments = partial(
                audalign.visrecognize.visrecognize_directory,
                against_directory=dir_or_list,
                start_end=target_start_end,
                img_width=img_width,
                volume_threshold=volume_threshold,
                volume_floor=volume_floor,
                vert_scaling=vert_scaling,
                horiz_scaling=horiz_scaling,
                calc_mse=ada_obj.calc_mse,
                max_lags=max_lags,
                use_multiprocessing=False,
                num_processes=1,
                plot=False,
                _file_audsegs=fine_aud_file_dict,
                _include_filename=True,
            )
        else:
            _calc_alignments = partial(
                audalign.correcognize.correcognize_directory,
                against_directory=dir_or_list,
                start_end=target_start_end,
                filter_matches=filter_matches,
                sample_rate=cor_sample_rate,
                locality=locality,
                locality_filter_prop=locality_filter_prop,
                max_lags=max_lags,
                _file_audsegs=fine_aud_file_dict,
                _include_filename=True,
                technique=technique,
                plot=False,
                use_multiprocessing=False,
                num_processes=1,
                **kwargs,
            )
        temp_file_list = []
        for file_path, _ in file_list:
            if os.path.basename(file_path) in ada_obj.file_names:
                temp_file_list += [file_path]

        with multiprocessing.Pool(ada_obj.num_processors) as pool:
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
            if name in ada_obj.file_names:
                if technique == "fingerprints":
                    alignment = audalign.recognize.recognize(
                        ada_obj,
                        file_path,
                        filter_matches=filter_matches,
                        locality=locality,
                        locality_filter_prop=locality_filter_prop,
                        start_end=None,
                        max_lags=max_lags,
                    )
                elif technique in ["correlation", "correlation_spectrogram"]:
                    alignment = audalign.correcognize.correcognize_directory(
                        file_path,
                        dir_or_list,
                        start_end=target_start_end,
                        filter_matches=filter_matches,
                        sample_rate=cor_sample_rate,
                        _file_audsegs=fine_aud_file_dict,
                        locality=locality,
                        locality_filter_prop=locality_filter_prop,
                        max_lags=max_lags,
                        technique=technique,
                        use_multiprocessing=ada_obj.get_multiprocessing(),
                        num_processes=ada_obj.get_num_processors(),
                        **kwargs,
                    )
                elif technique == "visual":
                    alignment = audalign.visrecognize.visrecognize_directory(
                        target_file_path=file_path,
                        against_directory=dir_or_list,
                        start_end=target_start_end,
                        img_width=img_width,
                        volume_threshold=volume_threshold,
                        volume_floor=volume_floor,
                        vert_scaling=vert_scaling,
                        horiz_scaling=horiz_scaling,
                        calc_mse=ada_obj.calc_mse,
                        max_lags=max_lags,
                        use_multiprocessing=ada_obj.get_multiprocessing(),
                        num_processes=ada_obj.get_num_processors(),
                        _file_audsegs=fine_aud_file_dict,
                    )
                file_names_and_paths[name] = file_path
                total_alignment[name] = alignment
    return total_alignment, file_names_and_paths


def calc_final_alignments(
    ada_obj,
    filename_list,
    file_dir,
    technique,
    total_alignment,
    destination_path,
    file_names_and_paths,
    write_extension,
    target_aligning,
    fine_aligning: bool,
):
    if ada_obj.alternate_strength_stat is not None:
        if technique == "fingerprints":
            ada_obj.alternate_strength_stat = ada_obj.CONFIDENCE
        elif technique == "visual":
            ada_obj.alternate_strength_stat = "ssim"
        else:
            ada_obj.alternate_strength_stat = ada_obj.CONFIDENCE
    files_shifts = find_most_matches(
        total_alignment, strength_stat=ada_obj.alternate_strength_stat
    )
    if not files_shifts:
        return
    files_shifts = find_matches_not_in_file_shifts(
        total_alignment, files_shifts, strength_stat=ada_obj.alternate_strength_stat
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
            ada_obj._write_shifted_files(
                files_shifts,
                destination_path,
                file_names_and_paths,
                write_extension,
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
        files_shifts[name] = file_match[audalign.Audalign.OFFSET_SECS][match_index]

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
                        nmatch_wt_most[main_name][audalign.Audalign.OFFSET_SECS] = None
                    if (
                        file_match[strength_stat][match_index]
                        > nmatch_wt_most[main_name]["match_strength"]
                    ):
                        nmatch_wt_most["match_strength"] = file_match[strength_stat][
                            match_index
                        ]
                        nmatch_wt_most[audalign.Audalign.OFFSET_SECS] = (
                            file_match[audalign.Audalign.OFFSET_SECS][match_index]
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
