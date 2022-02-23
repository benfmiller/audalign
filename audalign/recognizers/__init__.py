import os
import typing
from abc import ABC

import audalign.filehandler as filehandler
from audalign.config import BaseConfig


class BaseRecognizer(ABC):
    config: BaseConfig

    def __init__(self, config: BaseConfig = None):
        """takes a config object"""
        raise NotImplementedError

    def recognize(self, target_file: str, against: str) -> dict:
        """takes two files to recognize against

        returns a dict of results."""
        raise NotImplementedError

    def align_stat_print(self) -> None:
        """Status print during alignment

        used in fingerprinting to print total number of fingerprints"""
        pass

    def align_get_file_names(
        self,
        file_list: list,
        file_dir: typing.Optional[str],
        target_aligning: bool,
        fine_aud_file_dict: typing.Optional[dict],
    ) -> list:
        """returns a list of target basenames for use in alignment

        Args:
            file_list (typing.Union[str, list]): list of files
            file_dir (typing.Optional[str]): a directory
            target_aligning (bool): whether or not this a target alignment
            fine_aud_file_dict (typing.Optional[dict]): for fine aligning

        Returns:
            list: target file basenames
        """
        if target_aligning:
            file_names = [os.path.basename(x) for x in file_list]
        elif file_dir:
            file_names = filehandler.get_audio_files_directory(file_dir)
        elif fine_aud_file_dict:
            file_names = [os.path.basename(x) for x in fine_aud_file_dict.keys()]
        else:
            file_names = [os.path.basename(x) for x in file_list]
        return file_names

    def check_align_hook(
        self,
        file_list: list,
        dir_or_list: typing.Union[str, list],
        target_aligning: bool,
        fine_aud_file_dict: typing.Optional[dict],
    ) -> bool:
        """True if using a custom function rather than the regular alignment function

        Args:
            file_list (typing.Union[str, list]): list of files
            dir_or_list (typing.Union[str, list]): a directory or list of files
            target_aligning (bool): whether or not this a target alignment
            fine_aud_file_dict (typing.Optional[dict]): for fine aligning

        Returns:
            bool
        """
        return False

    def align_hook(
        self,
        file_list,
        dir_or_list,
        target_aligning: bool,
        fine_aud_file_dict: typing.Optional[dict],
    ):
        """Implement this if check align hook can return True
        gives a functools partial function to be given to a multiprocessing pool


        Args:
            file_list (typing.Union[str, list]): list of files
            dir_or_list (typing.Union[str, list]): a directory or list of files
            target_aligning (bool): whether or not this a target alignment
            fine_aud_file_dict (typing.Optional[dict]): for fine aligning

        Returns:
            functools partial function
        """
        raise NotImplementedError

    def align_post_hook(
        self,
        file_list,
        dir_or_list,
        target_aligning: bool,
        fine_aud_file_dict: typing.Optional[dict],
    ):
        """Overide if you want something to happen after each alignment

        Args:
            file_list (typing.Union[str, list]): list of files
            dir_or_list (typing.Union[str, list]): a directory or list of files
            target_aligning (bool): whether or not this a target alignment
            fine_aud_file_dict (typing.Optional[dict]): for fine aligning

        """
        pass

    def _align(
        self,
        file_path,
        dir_or_list,
    ):
        """Called directly by audalign's align methods"""
        raise NotImplementedError
