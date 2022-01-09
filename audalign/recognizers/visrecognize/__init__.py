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


class VisualRecognizer(BaseRecognizer):
    config: VisualConfig

    def __init__(self, config: VisualConfig = None):
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
        target_aligning: bool,
        fine_aud_file_dict: typing.Optional[dict],
    ):
        if self.config.multiprocessing == True and not target_aligning:
            return True
        return False

    def align_hook(
        self,
        file_list,
        dir_or_list,
        target_aligning: bool,
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
