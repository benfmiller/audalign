from audalign.recognizers import BaseRecognizer
from audalign.config.correlation import CorrelationConfig
from audalign.recognizers.correcognize.correcognize import (
    correcognize,
    correcognize_directory,
)

import os

# TODO fine align compatibility


class CorrelationRecognizer(BaseRecognizer):
    def __init__(self, config: CorrelationConfig = None):
        # super().__init__(config=config)
        self.config = CorrelationConfig() if config is None else config
        self.last_recognition = None

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

    # def correcognize_directory(
    #     self,
    #     target_file_path: str,
    #     against_directory: str,
    #     _file_audsegs: dict = None,
    #     **kwargs,
    # ):
    #     """Uses cross correlation to find alignment

    #     Faster than visrecognize or recognize and more useful for amplitude
    #     based alignments

    #     Args
    #     ----
    #         target_file_path (str): File to recognize
    #         against_directory (str): Directory to recognize against
    #         start_end (tuple(float, float), optional): Silences before and after start and end. (0, -1) Silences last second, (5.4, 0) silences first 5.4 seconds
    #         filter_matches (float, optional): Filters based on confidence. Ranges between 0 and 1. Defaults to 0.5.
    #         match_len_filter (int, optional): Limits number of matches returned. Defaults to 30.
    #         locality (float): filters matches to only count within locality. In seconds
    #         locality_filter_prop (int, float,optional): within each offset, filters locality tuples by proportion of highest confidence to tuple confidence
    #         sample_rate (int, optional): Decodes audio file to this sample rate. Defaults to fingerprint.DEFAULT_FS.
    #         max_lags (float, optional): Maximum lags in seconds.
    #         plot (bool, optional): Plots. Defaults to False.
    #         _file_audsegs (dict, optional): For use with align.
    #         kwargs: additional arguments for scipy.signal.find_peaks.

    #     Returns
    #     -------
    #         dict: dictionary of recognition information
    #     """
    #     return correcognize.correcognize_directory(
    #         target_file_path,
    #         against_directory,
    #         start_end=start_end,
    #         filter_matches=filter_matches,
    #         match_len_filter=match_len_filter,
    #         locality=locality,
    #         locality_filter_prop=locality_filter_prop,
    #         sample_rate=sample_rate,
    #         plot=plot,
    #         max_lags=max_lags,
    #         _file_audsegs=_file_audsegs,
    #         use_multiprocessing=self.multiprocessing,
    #         num_processes=self.num_processors,
    #         **kwargs,
    #     )

    # def correcognize(
    #     self,
    #     target_file_path: str,
    #     against_file_path: str,
    #     start_end_target: tuple = None,
    #     start_end_against: tuple = None,
    #     filter_matches: float = 0.5,
    #     match_len_filter: int = None,
    #     locality: float = None,
    #     locality_filter_prop: float = None,
    #     sample_rate: int = fingerprint.DEFAULT_FS,
    #     max_lags: float = None,
    #     plot: bool = False,
    #     **kwargs,
    # ):
    #     """Uses cross correlation to find alignment

    #     Faster than visrecognize or recognize and more useful for amplitude
    #     based alignments

    #     Args
    #     ----
    #         target_file_path (str): File to recognize
    #         against_file_path (str): File to recognize against
    #         start_end_target (tuple(float, float), optional): Silences before and after start and end. (0, -1) Silences last second, (5.4, 0) silences first 5.4 seconds
    #         start_end_against (tuple(float, float), optional): Silences before and after start and end. (0, -1) Silences last second, (5.4, 0) silences first 5.4 seconds
    #         filter_matches (float, optional): Filters based on confidence. Ranges between 0 and 1. Defaults to 0.0.
    #         match_len_filter (int, optional): Limits number of matches returned. Defaults to 30.
    #         locality (float): filters matches to only count within locality. In seconds
    #         locality_filter_prop (int, float,optional): within each offset, filters locality tuples by proportion of highest confidence to tuple confidence
    #         sample_rate (int, optional): Decodes audio file to this sample rate. Defaults to fingerprint.DEFAULT_FS.
    #         max_lags (float, optional): Maximum lags in seconds.
    #         plot (bool, optional): Plots. Defaults to False.
    #         kwargs: additional arguments for scipy.signal.find_peaks.

    #     Returns
    #     -------
    #         dict: dictionary of recognition information
    #     """
    #     return correcognize.correcognize(
    #         target_file_path,
    #         against_file_path,
    #         start_end_target=start_end_target,
    #         start_end_against=start_end_against,
    #         filter_matches=filter_matches,
    #         match_len_filter=match_len_filter,
    #         locality=locality,
    #         locality_filter_prop=locality_filter_prop,
    #         sample_rate=sample_rate,
    #         plot=plot,
    #         max_lags=max_lags,
    #         **kwargs,
    #     )
