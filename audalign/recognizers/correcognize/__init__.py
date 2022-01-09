from audalign import config
from audalign.recognizers import BaseRecognizer
from audalign.config.correlation import CorrelationConfig
from audalign.recognizers.correcognize.correcognize import (
    correcognize,
    correcognize_directory,
)

import os
import typing
import copy
from functools import partial


class CorrelationRecognizer(BaseRecognizer):
    """
    Uses cross correlation to find alignment

    Faster than visrecognize or recognize and more useful for amplitude
    based alignments
    """

    config: CorrelationConfig

    def __init__(self, config: CorrelationConfig = None):
        self.config = CorrelationConfig() if config is None else config

    def recognize(
        self,
        file_path: str,
        against_path: str,
    ):
        if os.path.isdir(file_path):
            raise ValueError(f"file_path {file_path} must be a file")
        if os.path.isdir(against_path):
            recognition = correcognize_directory(file_path, against_path, self.config)
        else:
            recognition = correcognize(file_path, against_path, self.config)
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
            correcognize_directory,
            against_directory=dir_or_list,
            config=temp_config,
            _file_audsegs=fine_aud_file_dict,
            _include_filename=True,
        )

    def _align(self, file_path, dir_or_list):
        return correcognize_directory(file_path, dir_or_list, self.config)
