import audalign.fingerprint as fingerprint
from audalign.filehandler import read, find_files
import scipy.signal as signal


def correcognize(
    target_file_path: str,
    against_file_path: str,
    start_end_target: tuple = None,
    start_end_against: tuple = None,
    filter_matches: int = 0,
    plot: bool = False,
):
    ...


def correcognize_directory(
    target_file_path: str,
    against_directory: str,
    start_end: tuple = None,
    filter_matches: int = 0,
    plot: bool = False,
):
    print(f"{target_file_path} : {against_directory}")
    ...