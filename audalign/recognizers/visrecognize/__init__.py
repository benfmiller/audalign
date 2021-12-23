from audalign import config
from audalign.recognizers import BaseRecognizer
from audalign.config.visual import VisualConfig
from audalign.recognizers.visrecognize.visrecognize import (
    visrecognize,
    visrecognize_directory,
)
import os
import typing
import copy
from functools import partial

# TODO fine align compatibility


class VisualRecognizer(BaseRecognizer):
    config: VisualConfig

    def __init__(self, config: VisualConfig = None):
        # super().__init__(config=config)
        self.config = VisualConfig() if config is None else config
        self.last_recognition = None

    def recognize(
        self,
        file_path: str,
        against_path: str,
    ):
        if os.path.isdir(file_path):
            raise ValueError(f"file_path {file_path} must be a file")
        if os.path.isdir(against_path):
            recognition = visrecognize_directory(file_path, against_path, self.config)
        else:
            recognition = visrecognize(file_path, against_path, self.config)
        self.last_recognition = recognition
        return recognition

    def check_align_hook(
        self,
        file_list,
        dir_or_list,
        max_lags: typing.Optional[float],
        target_aligning: bool,
        fine_aud_file_dict: typing.Optional[dict],
    ):
        if self.config.multiprocessing == True and not target_aligning:
            return True
        return False

    def align_hook(
        self,
        dir_or_list,
        fine_aud_file_dict: typing.Optional[dict],
    ):

        temp_config = copy.deepcopy(self.config)
        temp_config.multiprocessing = False
        temp_config.plot = False

        return partial(
            visrecognize_directory,
            against_directory=dir_or_list,
            config=temp_config,
            _file_audsegs=fine_aud_file_dict,
            _include_filename=True,
        )

    def _align(self, file_path, dir_or_list):
        return visrecognize_directory(file_path, dir_or_list, self.config)


# def visrecognize_directory(
#     self,
#     target_file_path: str,
#     against_directory: str,
#     start_end: tuple = None,
#     img_width: float = 1.0,
#     volume_threshold: float = 215.0,
#     volume_floor: float = 10.0,
#     vert_scaling: float = 1.0,
#     horiz_scaling: float = 1.0,
#     calc_mse: bool = False,
#     max_lags: float = None,
#     plot: bool = False,
#     _file_audsegs: dict = None,
# ) -> dict:
#     """Recognize target file against against directory visually.
#     Uses image processing similarity techniques to identify areas with similar spectrums.
#     Uses multiprocessing if multiprocessing variable is set to true
#     Uses audalign freq_threshold as well

#     Args
#     ----
#             target_file_path (str): File to recognize.
#             against_directory (str): Recognize against all files in directory.
#             start_end (tuple(float, float), optional): Silences before and after start and end. (0, -1) Silences last second, (5.4, 0) silences first 5.4 seconds
#             img_width (float): width of spectrogram image for recognition.
#             volume_threshold (int): doesn't find stats for sections with max volume below threshold.
#             volume_floor (float): ignores volume levels below floor.
#             vert_scaling (float): scales vertically to speed up calculations. Smaller numbers have smaller images.
#             horiz_scaling (float): scales horizontally to speed up calculations. Smaller numbers have smaller images. Affects alignment granularity.
#             max_lags (float, optional): Maximum lags in seconds.
#             calc_mse (bool): also calculates mse for each shift if true. If false, uses default mse 20000000
#             plot (bool): plot the spectrogram of each audio file.

#     Returns
#     -------
#     match_result : dict
#             dictionary containing match time and match info

#             or

#             None : if no match
#     """
#     return visrecognize.visrecognize_directory(
#         target_file_path,
#         against_directory,
#         start_end=start_end,
#         img_width=img_width,
#         volume_threshold=volume_threshold,
#         volume_floor=volume_floor,
#         vert_scaling=vert_scaling,
#         horiz_scaling=horiz_scaling,
#         calc_mse=calc_mse,
#         max_lags=max_lags,
#         use_multiprocessing=self.multiprocessing,
#         num_processes=self.num_processors,
#         plot=plot,
#         _file_audsegs=_file_audsegs,
#     )


# def visrecognize(
#     self,
#     target_file_path: str,
#     against_file_path: str,
#     start_end_target: tuple = None,
#     start_end_against: tuple = None,
#     img_width: float = 1.0,
#     volume_threshold: float = 215.0,
#     volume_floor: float = 10.0,
#     vert_scaling: float = 1.0,
#     horiz_scaling: float = 1.0,
#     calc_mse: bool = False,
#     max_lags: float = None,
#     plot: bool = False,
# ) -> dict:
#     """Recognize target file against against file visually.
#     Uses image processing similarity techniques to identify areas with similar spectrums.
#     Uses multiprocessing if multiprocessing variable is set to true
#     Uses audalign freq_threshold as well

#     Args
#     ----
#             target_file_path (str): File to recognize.
#             against_file_path (str): Recognize against.
#             start_end_target (tuple(float, float), optional): Silences before and after start and end. (0, -1) Silences last second, (5.4, 0) silences first 5.4 seconds
#             start_end_against (tuple(float, float), optional): Silences before and after start and end. (0, -1) Silences last second, (5.4, 0) silences first 5.4 seconds
#             img_width (float): width of spectrogram image for recognition.
#             volume_threshold (float): doesn't find stats for sections with max volume below threshold.
#             volume_floor (float): ignores volume levels below floor.
#             vert_scaling (float): scales vertically to speed up calculations. Smaller numbers have smaller images.
#             horiz_scaling (float): scales horizontally to speed up calculations. Smaller numbers have smaller images. Affects alignment granularity.
#             calc_mse (bool): also calculates mse for each shift if true. If false, uses default mse 20000000
#             max_lags (float, optional): Maximum lags in seconds.
#             plot (bool): plot the spectrogram of each audio file.

#     Returns
#     -------
#     match_result : dict
#             dictionary containing match time and match info

#             or

#             None : if no match
#     """
#     return visrecognize.visrecognize(
#         target_file_path,
#         against_file_path,
#         start_end_target=start_end_target,
#         start_end_against=start_end_against,
#         img_width=img_width,
#         volume_threshold=volume_threshold,
#         volume_floor=volume_floor,
#         vert_scaling=vert_scaling,
#         horiz_scaling=horiz_scaling,
#         calc_mse=calc_mse,
#         max_lags=max_lags,
#         use_multiprocessing=self.multiprocessing,
#         num_processes=self.num_processors,
#         plot=plot,
#     )
