import audalign.fingerprint as fingerprint
from audalign.filehandler import read, find_files, get_shifted_file
from pydub.exceptions import CouldntDecodeError
import tqdm
import time
import os
import sys
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from functools import partial
from PIL import Image

lower_clip = 5
upper_clip = 255

# ------------------------------------------------------------------------------------------


def visrecognize(
    target_file_path: str,
    against_file_path: str,
    start_end_target: tuple = None,
    start_end_against: tuple = None,
    img_width=1.0,
    volume_threshold=215.0,
    volume_floor: float = 10.0,
    vert_scaling: float = 1.0,
    horiz_scaling: float = 1.0,
    calc_mse=False,
    max_lags: float = None,
    use_multiprocessing=True,
    num_processes=None,
    plot=False,
):
    # With frequency of 44100
    # Each frame is 0.0929 seconds with an overlap ratio of .5,
    # so moving over one frame moves 0.046 seconds
    # 1 second of frames is 21.55 frames.
    #
    # add option to specify which value to sort by?
    # PSNR

    # volume_threshold -= volume_floor

    t = time.time()

    img_width = get_frame_width(img_width)

    target_arr2d, transposed_target_arr2d = get_arrays(
        target_file_path,
        volume_floor=volume_floor,
        vert_scaling=vert_scaling,
        horiz_scaling=horiz_scaling,
        start_end=start_end_target,
    )

    target_index_list = find_index_arr(
        transposed_target_arr2d, volume_threshold, img_width
    )

    against_arr2d, transposed_against_arr2d = get_arrays(
        against_file_path,
        volume_floor=volume_floor,
        vert_scaling=vert_scaling,
        horiz_scaling=horiz_scaling,
        start_end=start_end_against,
    )
    results_list = _visrecognize(
        transposed_target_arr2d=transposed_target_arr2d,
        target_file_path=target_file_path,
        target_index_list=target_index_list,
        against_file_path=against_file_path,
        transposed_against_arr2d=transposed_against_arr2d,
        img_width=img_width,
        volume_threshold=volume_threshold,
        calc_mse=calc_mse,
        use_multiprocessing=use_multiprocessing,
        num_processes=num_processes,
        max_lags=max_lags,
    )
    file_match = process_results(
        results_list,
        os.path.basename(against_file_path),
        horiz_scaling=horiz_scaling,
    )
    t = time.time() - t

    if plot:
        plot_two_images(
            target_arr2d,
            against_arr2d,
            imgA_title=os.path.basename(target_file_path),
            imgB_title=os.path.basename(against_file_path),
        )

    result = {}

    if len(file_match) > 0:
        result["match_time"] = t
        result["match_info"] = file_match
        return result

    return None


# ------------------------------------------------------------------------------------------


def _visrecognize_directory(
    file_path: str,
    target_file_path: str,
    start_end: tuple = None,
    img_width=1.0,
    volume_threshold=215.0,
    volume_floor: float = 10.0,
    vert_scaling: float = 1.0,
    horiz_scaling: float = 1.0,
    calc_mse: bool = False,
    max_lags: float = None,
    use_multiprocessing: bool = True,
    num_processes: int = None,
    plot: bool = False,
    _file_audsegs: dict = None,
):
    if type(target_file_path) == str:
        target_arr2d, transposed_target_arr2d = get_arrays(
            target_file_path,
            volume_floor=volume_floor,
            vert_scaling=vert_scaling,
            horiz_scaling=horiz_scaling,
            start_end=start_end,
            _file_audsegs=_file_audsegs,
        )

        target_index_list = find_index_arr(
            transposed_target_arr2d, volume_threshold, img_width
        )
    else:
        (
            target_file_path,
            target_arr2d,
            transposed_target_arr2d,
            target_index_list,
        ) = target_file_path

    file_path, _ = file_path

    if os.path.basename(file_path) == os.path.basename(target_file_path):
        return {}
    try:
        against_arr2d, transposed_against_arr2d = get_arrays(
            file_path,
            volume_floor=volume_floor,
            vert_scaling=vert_scaling,
            horiz_scaling=horiz_scaling,
            _file_audsegs=_file_audsegs,
        )
        results_list = _visrecognize(
            transposed_target_arr2d=transposed_target_arr2d,
            target_file_path=target_file_path,
            target_index_list=target_index_list,
            against_file_path=file_path,
            transposed_against_arr2d=transposed_against_arr2d,
            img_width=img_width,
            volume_threshold=volume_threshold,
            calc_mse=calc_mse,
            use_multiprocessing=use_multiprocessing,
            num_processes=num_processes,
            max_lags=max_lags,
        )
        single_file_match = process_results(
            results_list,
            os.path.basename(file_path),
            horiz_scaling=horiz_scaling,
        )
        if plot:
            plot_two_images(
                target_arr2d,
                against_arr2d,
                imgA_title=os.path.basename(target_file_path),
                imgB_title=os.path.basename(file_path),
            )
        return single_file_match
    except CouldntDecodeError:
        print(f'File "{file_path}" could not be decoded')
        return {}


def visrecognize_directory(
    target_file_path: str,
    against_directory: str,
    start_end: tuple = None,
    img_width=1.0,
    volume_threshold=215.0,
    volume_floor: float = 10.0,
    vert_scaling: float = 1.0,
    horiz_scaling: float = 1.0,
    calc_mse: bool = False,
    max_lags: float = None,
    use_multiprocessing: bool = True,
    num_processes: int = None,
    plot: bool = False,
    _file_audsegs: dict = None,
    _include_filename=False,
):
    # With frequency of 44100
    # Each frame is 0.0929 seconds with an overlap ratio of .5,
    # so moving over one frame moves 0.046 seconds
    # 1 second of frames is 21.55 frames.
    #
    # add option to specify which value to sort by?
    # PSNR

    # volume_threshold -= volume_floor

    t = time.time()

    img_width = get_frame_width(img_width)

    if use_multiprocessing == False:
        target_arr2d, transposed_target_arr2d = get_arrays(
            target_file_path,
            volume_floor=volume_floor,
            vert_scaling=vert_scaling,
            horiz_scaling=horiz_scaling,
            start_end=start_end,
            _file_audsegs=_file_audsegs,
        )

        target_index_list = find_index_arr(
            transposed_target_arr2d, volume_threshold, img_width
        )

        target_file_path = (
            target_file_path,
            target_arr2d,
            transposed_target_arr2d,
            target_index_list,
        )

    if type(against_directory) == str:
        against_files = list(find_files(against_directory))
    else:
        against_files = list(zip(against_directory, ["_"] * len(against_directory)))

    _visrecognize_directory_ = partial(
        _visrecognize_directory,
        target_file_path=target_file_path,
        start_end=start_end,
        img_width=img_width,
        volume_threshold=volume_threshold,
        volume_floor=volume_floor,
        vert_scaling=vert_scaling,
        horiz_scaling=horiz_scaling,
        calc_mse=calc_mse,
        max_lags=max_lags,
        use_multiprocessing=False,
        num_processes=num_processes,
        plot=plot,
        _file_audsegs=_file_audsegs,
    )

    if use_multiprocessing == False:
        results_list = []
        for file_path in against_files:
            results_list += [_visrecognize_directory_(file_path)]
    else:
        try:
            nprocesses = num_processes or multiprocessing.cpu_count()
        except NotImplementedError:
            nprocesses = 1
        else:
            nprocesses = 1 if nprocesses <= 0 else nprocesses

        with multiprocessing.Pool(nprocesses) as pool:
            results_list = pool.map(
                _visrecognize_directory_, tqdm.tqdm(list(against_files))
            )
            pool.close()
            pool.join()

    file_match = {}
    for i in results_list:
        file_match = {**file_match, **i}

    t = time.time() - t

    result = {}
    if len(file_match) > 0:
        result["match_time"] = t
        result["match_info"] = file_match
        if _include_filename:
            if type(target_file_path) == tuple:
                result["filename"] = target_file_path[0]
            else:
                result["filename"] = target_file_path
        return result

    return None


# ------------------------------------------------------------------------------------------


def _visrecognize(
    transposed_target_arr2d,
    target_file_path: str,
    target_index_list: list,
    against_file_path: str,
    transposed_against_arr2d,
    img_width=1.0,
    volume_threshold=215.0,
    calc_mse=False,
    use_multiprocessing=True,
    num_processes=None,
    max_lags: float = None,
):

    # th, _ = transposed_target_arr2d.shape
    # ah, _ = transposed_against_arr2d.shape

    # plot_two_images(transposed_target_arr2d, transposed_against_arr2d)

    # print(f"Target height: {th}, target width:")
    # print(f"against height: {ah}")
    # print(f"length of target: {len(transposed_target_arr2d)}")
    print(
        f"Comparing {os.path.basename(target_file_path)} against {os.path.basename(against_file_path)}... "
    )

    # create index list
    against_index_list = find_index_arr(
        transposed_against_arr2d, volume_threshold, img_width
    )

    index_list = pair_index_tuples(
        target_index_list, against_index_list, max_lags=max_lags
    )

    # offsets = [x[0] - x[1] for x in index_list]
    # print()
    # print(offsets.count(215))

    _calculate_comp_values = partial(
        calculate_comp_values,
        img_width=img_width,
        calc_mse=calc_mse,
    )

    # calculate all mse and ssim values
    if use_multiprocessing == True and sys.platform != "linux":

        try:
            nprocesses = num_processes or multiprocessing.cpu_count()
        except NotImplementedError:
            nprocesses = 1
        else:
            nprocesses = 1 if nprocesses <= 0 else nprocesses

        index_list = divy_index_list(
            index_list, transposed_target_arr2d, transposed_against_arr2d, nprocesses
        )

        with multiprocessing.Pool(nprocesses) as pool:
            results_list = pool.map(_calculate_comp_values, tqdm.tqdm(list(index_list)))
            pool.close()
            pool.join()
    else:
        index_list = divy_index_list(
            index_list, transposed_target_arr2d, transposed_against_arr2d, 1
        )
        results_list = []
        for i in list(index_list):
            results_list += [_calculate_comp_values(i)]
    new_results_list = []
    for i in results_list:
        new_results_list += i

    return new_results_list


# ------------------------------------------------------------------------------------------


def divy_index_list(index_list, target_arr, against_arr, nprocesses):
    sublists = [list(index_list)[i::nprocesses] for i in range(nprocesses)]
    return zip(sublists, [target_arr] * nprocesses, [against_arr] * nprocesses)


def get_arrays(
    file_path: str,
    volume_floor: float = 10.0,
    vert_scaling: float = 1.0,
    horiz_scaling: float = 1.0,
    start_end: tuple = None,
    _file_audsegs: dict = None,
):
    if _file_audsegs is not None:
        samples = get_shifted_file(file_path, _file_audsegs[file_path])
    else:
        samples, _ = read(file_path, start_end=start_end)
    arr2d = fingerprint.fingerprint(samples, retspec=True)
    if fingerprint.threshold > 0:
        arr2d = arr2d[0 : -fingerprint.threshold]
    arr2d = np.clip(arr2d, volume_floor, upper_clip)
    if vert_scaling != 1.0 or horiz_scaling != 1.0:
        array_image = Image.fromarray(np.uint8(arr2d))
        array_image = array_image.resize(
            (
                int(array_image.size[0] * horiz_scaling),
                int(array_image.size[1] * vert_scaling),
            ),
            Image.NEAREST,
        )
        arr2d = np.array(array_image)

    # arr2d -= volume_floor
    transposed_arr2d = np.transpose(arr2d)
    # transposed_arr2d -= volume_floor
    return arr2d, transposed_arr2d


# ------------------------------------------------------------------------------------------


def get_frame_width(seconds_width: float):
    return max(
        int(
            seconds_width
            // (
                fingerprint.DEFAULT_WINDOW_SIZE
                / fingerprint.DEFAULT_FS
                * fingerprint.DEFAULT_OVERLAP_RATIO
            )
        ),
        1,
    )


# ------------------------------------------------------------------------------------------


def find_index_arr(arr2d, threshold, img_width):
    index_list = []
    for i in range(0, len(arr2d) - img_width):
        if np.amax(arr2d[i : i + img_width]) >= threshold:
            index_list += [i]
    return index_list


def pair_index_tuples(target_list, against_list, max_lags: float = None):
    index_pairs = []
    if max_lags is None:
        for i in target_list:
            for j in against_list:
                index_pairs += [(i, j)]
    else:
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
        for i in target_list:
            for j in against_list:
                if abs(j - i) <= max_lags:
                    index_pairs += [(i, j)]
    return index_pairs


# ------------------------------------------------------------------------------------------


def calculate_comp_values(
    index_tuple_target_arr_against_arr, img_width=0, calc_mse=False
):
    # print(np.amax(target_arr2d[index_tuple[0] : index_tuple[0] + img_width]))
    # array.mean() very small range of values, usually between 0.4 and 2
    # Plus, finding the max only uses regions with large peaks, which could reduce
    # noisy secions being included.
    index_tuples = index_tuple_target_arr_against_arr[0]
    target_arr2d = index_tuple_target_arr_against_arr[1]
    against_arr2d = index_tuple_target_arr_against_arr[2]
    results_list = []
    for index_tuple in tqdm.tqdm(index_tuples):
        try:
            if calc_mse:
                m = mean_squared_error(
                    target_arr2d[index_tuple[0] : index_tuple[0] + img_width],
                    against_arr2d[index_tuple[1] : index_tuple[1] + img_width],
                )
            else:
                m = 20000000
            s = ssim(
                target_arr2d[index_tuple[0] : index_tuple[0] + img_width],
                against_arr2d[index_tuple[1] : index_tuple[1] + img_width],
            )
            results_list += [(index_tuple[1], index_tuple[0], (m, s))]
        except ZeroDivisionError as e:
            m = 10000000
            print(
                f"zero division error for index {index_tuple} and img width{img_width}"
            )
            s = 10000000
            results_list += [(index_tuple[1], index_tuple[0], (m, s))]
    return results_list


# ------------------------------------------------------------------------------------------


def process_results(results_list, filename, horiz_scaling: float = 1.0):
    """processes results from recognition and returns a pretty json
    If you want to mess with the weighting of num matches vs score, this is the place to do it.
    Current weighting seems to work the best, though.

    Args:
        results_list ([type]): [description]
        filename ([type]): [description]
        horiz_scaling (float, optional): [description]. Defaults to 1.0.

    Returns:
        [type]: [description]
    """
    print("Calculating results... ", end="")
    i = 0  # remove bad results or results below threshold
    while i < len(results_list):
        if results_list[i][2][0] == 10000000 or results_list[i][2][1] == 10000000:
            results_list.pop(i)
            continue
        i += 1

    offset_dict = {}  # aggregate results by time difference
    for i in results_list:
        if i[0] - i[1] not in offset_dict:
            offset_dict[i[0] - i[1]] = [0, 0, 0]  # mse,ssim,total
        temp_result = offset_dict[i[0] - i[1]]
        temp_result[0] += i[2][0]
        temp_result[1] += i[2][1]
        temp_result[2] += 1
        offset_dict[i[0] - i[1]] = temp_result

    for i in offset_dict.keys():  # average mse and ssim
        temp_result = offset_dict[i]
        temp_result[0] /= temp_result[2]
        temp_result[1] /= temp_result[2]
        offset_dict[i] = temp_result

    match_offsets = []
    for t_difference, match_data in offset_dict.items():
        match_offsets.append((match_data, t_difference))
    match_offsets = sorted(
        match_offsets,
        reverse=True,
        key=lambda x: (x[0][1], x[0][2]),
        # key=lambda x: (x[0][2], x[0][1]),
        # key=lambda x: (np.log2(x[0][2] + 1) * (np.log(x[0][1] + 100) / np.log(2.5))),
    )  # sort by ssim must be reversed for ssim
    # match_offsets, reverse=True, key=lambda x: (x[0][2], x[0][1])
    # match_offsets, reverse=True, key=lambda x: x[0][2] sorts by num matches
    # match_offsets, reverse=True, key=lambda x: (x[0][2], x[0][1]) sorts once by ssim, then finally by num matches
    # math.log

    offset_count = []
    offset_diff = []
    offset_ssim = []
    offset_mse = []
    for i in match_offsets:
        offset_count.append(i[0][2])
        offset_diff.append(i[1])
        offset_ssim.append(i[0][1])
        offset_mse.append(i[0][0])

    match = {}
    match[filename] = {}

    match[filename]["num_matches"] = offset_count
    match[filename]["offset_frames"] = offset_diff
    match[filename]["ssim"] = offset_ssim
    match[filename]["mse"] = offset_mse

    offset_seconds = []
    for i in offset_diff:
        nseconds = round(
            float(i)
            / fingerprint.DEFAULT_FS
            * fingerprint.DEFAULT_WINDOW_SIZE
            * fingerprint.DEFAULT_OVERLAP_RATIO
            / horiz_scaling,
            5,
        )
        offset_seconds.append(nseconds)

    match[filename]["offset_seconds"] = offset_seconds
    print("done")

    if len(match[filename]["offset_seconds"]) > 0:
        return match
    return {}


# ------------------------------------------------------------------------------------------


def plot_two_images(
    imageA,
    imageB,
    title="Comparison",
    imgA_title=None,
    imgB_title=None,
    mse=None,
    ssim_value=None,
):
    # setup the figure
    fig = plt.figure(title)
    if mse or ssim_value:
        plt.suptitle(f"MSE: {mse:.4f}, SSIM: {ssim_value:.4f}")
    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA)  # , cmap=plt.cm.gray)
    plt.gca().invert_yaxis()
    if imgA_title:
        plt.title(imgA_title)
    # plt.axis("off")
    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB)  # , cmap=plt.cm.gray)
    plt.gca().invert_yaxis()
    if imgB_title:
        plt.title(imgB_title)
    # plt.axis("off")
    # show the images
    plt.show()
