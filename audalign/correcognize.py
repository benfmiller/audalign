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
    sample_rate: int = fingerprint.DEFAULT_FS,
    plot: bool = False,
):
    assert (
        sample_rate < 200000
    )  # I accidentally used 441000 once... not good, big crash

    target_array, _ = read(
        target_file_path, start_end=start_end_target, sample_rate=sample_rate
    )
    against_array, _ = read(
        against_file_path, start_end=start_end_against, sample_rate=sample_rate
    )

    sos = signal.butter(10, fingerprint.threshold, "hp", fs=sample_rate, output="sos")
    # sos = signal.butter(10, 0.125, "hp", fs=sample_rate, output="sos")
    target_array = signal.sosfilt(sos, target_array)
    against_array = signal.sosfilt(sos, against_array)
    correlation = signal.correlate(target_array, against_array)
    # correlation = target_array
    if plot:
        plot_cor(
            array_a=target_array,
            array_b=against_array,
            corr_array=correlation,
            sample_rate=sample_rate,
            arr_a_title=target_file_path,
            arr_b_title=against_file_path,
        )
    ...


def correcognize_directory(
    target_file_path: str,
    against_directory: str,
    start_end: tuple = None,
    filter_matches: int = 0,
    sample_rate: int = fingerprint.DEFAULT_FS,
    plot: bool = False,
):
    print(f"{target_file_path} : {against_directory}")
    ...


def process_results():
    # xin, fs = sf.read('recording1.wav')
    # frame_len = int(fs*5*1e-3)
    # dim_x =xin.shape
    # M = dim_x[0] # No. of rows
    # N= dim_x[1] # No. of col
    # sample_lim = frame_len*100
    # tau = [0]
    # M_lim = 20000 # for testing as processing takes time
    # for i in range(1,N):
    #     c = np.correlate(xin[0:M_lim,0],xin[0:M_lim,i],"full")
    #     maxlags = M_lim-1
    #     c = c[M_lim -1 -maxlags: M_lim + maxlags]
    #     Rmax_pos = np.argmax(c)
    #     pos = Rmax_pos-M_lim+1
    #     tau.append(pos)
    # print(tau)
    ...


# ---------------------------------------------------------------------------------


def plot_cor(
    array_a,
    array_b,
    corr_array,
    sample_rate,
    title="Comparison",
    arr_a_title=None,
    arr_b_title=None,
):
    new_vis_wsize = int(fingerprint.DEFAULT_WINDOW_SIZE / 44100 * sample_rate)
    fig = plt.figure(title)

    fig.add_subplot(3, 2, 1)
    plt.plot(array_a)
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    if arr_a_title:
        plt.title(arr_a_title)

    arr2d_a = fingerprint.fingerprint(
        array_a, fs=sample_rate, wsize=new_vis_wsize, retspec=True
    )
    fig.add_subplot(3, 2, 2)
    plt.imshow(arr2d_a)  # , cmap=plt.cm.gray)
    plt.gca().invert_yaxis()
    if arr_a_title:
        plt.title(arr_a_title)

    fig.add_subplot(3, 2, 3)
    plt.plot(array_b)
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    if arr_b_title:
        plt.title(arr_b_title)

    arr2d_b = fingerprint.fingerprint(
        array_b, fs=sample_rate, wsize=new_vis_wsize, retspec=True
    )
    fig.add_subplot(3, 2, 4)
    plt.imshow(arr2d_b)
    plt.gca().invert_yaxis()
    if arr_b_title:
        plt.title(arr_b_title)

    fig.add_subplot(3, 2, 5)
    plt.plot(corr_array)
    plt.title(f"Correlation")
    plt.xlabel("Sample Index")
    plt.ylabel("Offset")

    fig.tight_layout()

    plt.show()