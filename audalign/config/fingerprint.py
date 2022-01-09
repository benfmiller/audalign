from audalign.config import BaseConfig
import typing


class FingerprintConfig(BaseConfig):
    """
    Set the hash style and accuracy with the given setter methods

    hash style has four options. All fingerprints must be of the same hash style to match.

    'base' hash style consists of two peaks. Two frequencies and a time difference.
    Creates many matches but is insensitive to noise.

    'panako' hash style consists of three peaks. Two differences in frequency, two frequency
    bands, one time difference ratio. Creates few matches, very resistant to noise.

    'panako_mod' hash style consists of three peaks. Two differences in frequency and one
    time difference ratio. Creates less matches than base, more than panako. moderately
    resistant to noise

    'base_three' hash style consists of three peaks. Three frequencies and two time differences.

    multiprocessing is set to True by default

    There are four accuracy levels with 1 being the lowest accuracy but the fastest. 3 is the highest recommended.
    4 gives the highest accuracy, but can take several gigabytes of memory for a couple files.
    Accuracy settings are acheived by manipulations in fingerprinting variables.
    """

    hash_style = "panako_mod"
    _accuracy = 2
    filter_matches = 1
    locality: typing.Optional[float] = None
    locality_filter_prop: typing.Optional[float] = None
    extensions = ["*"]
    _IDX_FREQ_I = 0
    _IDX_TIME_J = 1

    ######################################################################
    # Sampling rate, related to the Nyquist conditions, which affects
    # the range frequencies we can detect.
    sample_rate = 44100

    ######################################################################
    # Size of the FFT window, affects frequency granularity
    # Which is 0.0929 seconds
    fft_window_size = 4096

    ######################################################################
    # Ratio by which each sequential window overlaps the last and the
    # next window. Higher overlap will allow a higher granularity of offset
    # matching, but potentially more fingerprints.
    DEFAULT_OVERLAP_RATIO = 0.5

    ######################################################################
    # Degree to which a fingerprint can be paired with its neighbors --
    # higher will cause more fingerprints, but potentially better accuracy.
    default_fan_value = 15

    ######################################################################
    # Minimum amplitude in spectrogram in order to be considered a peak.
    # This can be raised to reduce number of fingerprints, but can negatively
    # affect accuracy.
    # 50 roughly cuts number of fingerprints in half compared to 0
    default_amp_min = 65

    ######################################################################
    # Number of cells around an amplitude peak in the spectrogram in order
    # for audalign to consider it a spectral peak. Higher values mean less
    # fingerprints and faster matching, but can potentially affect accuracy.
    peak_neighborhood_size = 20

    ######################################################################
    # Thresholds on how close or far fingerprints can be in time in order
    # to be paired as a fingerprint. If your max is too low, higher values of
    # default_fan_value may not perform as expected.
    min_hash_time_delta = 10
    max_hash_time_delta = 200

    ######################################################################
    # If True, will sort peaks temporally for fingerprinting;
    # not sorting will cut down number of fingerprints, but potentially
    # affect performance.
    peak_sort = True

    ######################################################################
    # Number of bits to grab from the front of the SHA1 hash in the
    # fingerprint calculation. The more you grab, the more memory storage,
    # with potentially lesser collisions of matches.
    FINGERPRINT_REDUCTION = 20

    rankings_minus = (
        (0.95, 4),
        (0.9, 3),
        (0.85, 2),
        (0.8, 1),
        (0.7, 0),
        (0.6, -1),
        (0.45, -2),
        (0.0, 0),
    )
    rankings_locality_top_match_tups = (
        (100, 10),
        (80, 9),
        (60, 8),
        (40, 7),
        (30, 6),
        (15, 5),
        (12, 4),
        (8, 3),
        (0, 1),
    )
    rankings_no_locality_top_match_tups = (
        (500, 10),
        (200, 9),
        (100, 8),
        (70, 7),
        (30, 6),
        (15, 5),
        (10, 4),
        (0, 1),
    )

    def set_hash_style(self, hash_style: str) -> None:
        """Sets the hash style. Must be one of ["base", "panako", "panako_mod", "base_three"]

        Args
        ----
            hash_style (str): Method to use for hashing of fingerprints
        """
        if hash_style not in ["base", "panako", "panako_mod", "base_three"]:
            raise ValueError(
                f'Hash style "{hash_style}" must be one of ["base", "panako", "panako_mod", "base_three"]'
            )
        self.hash_style = hash_style

    def get_hash_style(self) -> str:
        """Gets the hash style. Is one of ["base", "panako", "panako_mod", "base_three"]"""
        return self.hash_style

    def set_accuracy(self, accuracy: int) -> None:
        """
        Sets the accuracy level of audalign object

        There are four accuracy levels with 1 being the lowest accuracy but the fastest. 3 is the highest recommended.
        4 gives the highest accuracy, but can take several gigabytes of memory for a couple files.
        Accuracy settings are acheived by manipulations in fingerprinting variables.

        Specific values for accuracy levels were chosen semi-arbitrarily from experimentation to give a few good options.

        Args
        ----
            accuracy (int): which accuracy level: 1-4
        """
        accuracy = int(accuracy)
        if accuracy < 1 or accuracy > 4:
            raise ValueError(f"Accuracy '{accuracy}' must be between 1 and 4")
        self._accuracy = accuracy
        self._set_accuracy(accuracy)

    def _set_accuracy(self, accuracy: int) -> None:
        if accuracy == 1:
            self.default_fan_value = 15
            self.default_amp_min = 80
            self.min_hash_time_delta = 10
            self.max_hash_time_delta = 200
            self.peak_sort = True
        elif accuracy == 2:
            self.default_fan_value = 15
            self.default_amp_min = 65
            self.min_hash_time_delta = 10
            self.max_hash_time_delta = 200
            self.peak_sort = True
        elif accuracy == 3:
            self.default_fan_value = 40
            self.default_amp_min = 60
            self.min_hash_time_delta = 1
            self.max_hash_time_delta = 400
            self.peak_sort = True
        elif accuracy == 4:
            self.default_fan_value = 60
            self.default_amp_min = 55
            self.min_hash_time_delta = 1
            self.max_hash_time_delta = 2000
            self.peak_sort = True

    def get_accuracy(self) -> int:
        """Current Accuracy from 1-4

        Returns:
            [int]: Accuracy level
        """
        return self._accuracy
