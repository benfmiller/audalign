"""The numpy reading path must give exactly what the audioop / pydub path gives."""

import audioop

import numpy as np
import pytest
from pydub import AudioSegment, effects

from audalign import filehandler, pcm

RATES = [1000, 7999, 8000, 11025, 16000, 22050, 44056, 44100, 47999, 48000, 96000]

test_file = "test_audio/testers/test.mp3"
test_file2 = "test_audio/testers/pink_noise.mp3"
test_file_eig = "test_audio/test_shifts/Eigen-20sec.mp3"


def random_frames(rng, n, channels):
    kind = rng.integers(0, 4)
    if kind == 0:
        frames = rng.integers(-32768, 32768, size=(n, channels))
    elif kind == 1:  # extremes
        frames = rng.choice(np.array([-32768, -32767, -1, 0, 1, 32766, 32767]), size=(n, channels))
    elif kind == 2:  # quiet
        frames = rng.integers(-40, 40, size=(n, channels))
    else:  # a tone with noise
        t = np.arange(n)[:, None]
        frames = np.round(20000 * np.sin(t * rng.uniform(0.001, 0.5) + np.arange(channels)))
        frames = np.clip(frames + rng.integers(-500, 500, (n, channels)), -32768, 32767)
    return np.ascontiguousarray(frames.astype(np.int16))


def cases(seed, count):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        channels = int(rng.integers(1, 3))
        n = int(rng.choice([0, 1, 2, 3, 7, 100, 1000, int(rng.integers(1, 20000))]))
        yield rng, random_frames(rng, n, channels), channels


class TestPcm:
    def test_ratecv_matches_audioop(self):
        for rng, frames, channels in cases(1, 300):
            inrate, outrate = (int(r) for r in rng.choice(RATES, 2))
            if rng.random() < 0.2:
                inrate, outrate = (int(r) for r in rng.integers(1000, 200000, 2))
            if not len(frames):
                continue
            expected, _ = audioop.ratecv(frames.tobytes(), 2, channels, inrate, outrate, None)
            assert pcm.ratecv(frames, inrate, outrate).tobytes() == expected, (len(frames), channels, inrate, outrate)

    def test_tomono_matches_audioop(self):
        for _, frames, channels in cases(2, 300):
            if channels == 2:
                assert pcm.tomono(frames).tobytes() == audioop.tomono(frames.tobytes(), 2, 0.5, 0.5)

    def test_mul_matches_audioop(self):
        for rng, frames, _ in cases(3, 300):
            samples = frames[:, 0].copy()
            for factor in (0.0, 1.0, float(rng.uniform(0, 4)), float(rng.uniform(0.99, 1.01)), 60000.0):
                assert pcm.mul(samples, factor).tobytes() == audioop.mul(samples.tobytes(), 2, factor), factor

    @pytest.mark.parametrize("normalize", [True, False])
    def test_to_mono_16bit_matches_pydub(self, normalize):
        for rng, frames, channels in cases(4, 150):
            inrate = int(rng.choice(RATES))
            segment = AudioSegment(data=frames.tobytes(), sample_width=2, frame_rate=inrate, channels=channels)
            for sample_rate in (8000, int(rng.choice(RATES))):
                expected = segment.set_frame_rate(sample_rate).set_sample_width(2).set_channels(1)
                if normalize:
                    expected = effects.normalize(expected)
                result = filehandler._to_mono_16bit(segment, sample_rate, normalize)
                assert vars(result) == vars(expected), (len(frames), channels, inrate, sample_rate)


class TestFromFile:
    @pytest.mark.parametrize("file_path", [test_file, test_file2, test_file_eig])
    def test_matches_pydub(self, file_path):
        assert vars(filehandler._from_file(file_path)) == vars(AudioSegment.from_file(file_path))

    @pytest.mark.parametrize("sample_width, channels", [(2, 1), (2, 2), (1, 2), (3, 1), (4, 2)])
    def test_matches_pydub_for_wav(self, tmp_path, sample_width, channels):
        # 16-bit PCM is decoded without ffprobe; the other widths go through pydub.
        segment = AudioSegment.from_file(test_file)[:2000].set_channels(channels).set_sample_width(sample_width)
        file_path = str(tmp_path / "audio.wav")
        segment.export(file_path, format="wav")
        assert vars(filehandler._from_file(file_path)) == vars(AudioSegment.from_file(file_path))
