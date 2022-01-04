from abc import ABC
from audalign.config import BaseConfig
import audalign.filehandler as filehandler
import os
import typing


class BaseRecognizer(ABC):
    config: BaseConfig

    def __init__(self, config: BaseConfig = None):
        """takes a config object"""
        raise NotImplementedError

    def recognize(self, target_file: str, against: str) -> dict:
        """this recognizes"""
        raise NotImplementedError

    def align_stat_print(self) -> None:
        """Status print during alignment"""
        pass

    def align_get_file_names(
        self,
        filename_list: typing.Union[str, list],
        file_dir: typing.Optional[str],
        target_aligning: bool,
        fine_aud_file_dict: typing.Optional[dict],
    ) -> list:
        if target_aligning:
            file_names = [os.path.basename(x) for x in filename_list]
        elif file_dir:
            file_names = filehandler.get_audio_files_directory(file_dir)
        elif fine_aud_file_dict:
            file_names = [os.path.basename(x) for x in fine_aud_file_dict.keys()]
        else:
            file_names = [os.path.basename(x) for x in filename_list]
        return file_names

    def check_align_hook(
        self,
        file_list,
        dir_or_list,
        target_aligning: bool,
        fine_aud_file_dict: typing.Optional[dict],
    ):
        return False

    def align_hook(
        self,
        file_list,
        dir_or_list,
        target_aligning: bool,
        fine_aud_file_dict: typing.Optional[dict],
    ):
        """Implement this if check align hook can return True"""
        raise NotImplementedError

    def align_post_hook(
        self,
        file_list,
        dir_or_list,
        target_aligning: bool,
        fine_aud_file_dict: typing.Optional[dict],
    ):
        pass

    def _align(
        self,
        file_path,
        dir_or_list,
    ):
        """Called directly by audalign's align methods"""
        raise NotImplementedError
