import audalign.fingerprint as fingerprint
from audalign.filehandler import read, find_files
import scipy.signal as signal
import matplotlib.pyplot as plt


def correcognize(
    target_file_path: str,
    against_file_path: str,
    start_end_target: tuple = None,
    start_end_against: tuple = None,
    filter_matches: int = 0,
    plot: bool = False,
):

    target_array = read(target_file_path)

    if plot:
        plot_cor(
            array_a=target_array,
            array_b=against_array,
            corr_array=correlation,
            arr_a_title=target_file_path,
            arr_b_title=against_file_path,
        )
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


def plot_cor(
    array_a,
    array_b,
    corr_array,
    title="Comparison",
    arr_a_title=None,
    arr_b_title=None,
):
    fig = plt.figure(title)

    fig.add_subplot(3, 1, 1)
    plt.plot(array_a)
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    # plt.gca().invert_yaxis()
    if arr_a_title:
        plt.title(arr_a_title)

    fig.add_subplot(3, 1, 2)
    plt.plot(array_b)
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    # plt.gca().invert_yaxis()
    if arr_b_title:
        plt.title(arr_b_title)

    fig.add_subplot(3, 1, 3)
    plt.plot(corr_array)
    plt.title(f"Correlation")
    plt.xlabel("Sample Index")
    plt.ylabel("Offset")
    plt.show()