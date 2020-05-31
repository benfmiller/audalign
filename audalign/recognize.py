# encoding: utf-8
import audalign.fingerprint as fingerprint
import audalign.decoder as decoder
import numpy as np
import pyaudio
import time


class BaseRecognizer(object):
    def __init__(self, audalign):
        self.audalign = audalign
        self.Fs = fingerprint.DEFAULT_FS

    def _recognize(self, *channels_samples):
        matches = []
        for channel in channels_samples:
            matches.extend(self.audalign.find_matches(channel, Fs=self.Fs))
        return self.audalign.align_matches(matches)

    def recognize(self):
        pass  # base class does nothing


class FileRecognizer(BaseRecognizer):
    def __init__(self, audalign):
        super(FileRecognizer, self).__init__(audalign)

    def recognize_file(self, file_path):
        try:
            channels_samples, self.Fs, file_hash = decoder.read(
                file_path, limit=self.audalign.limit
            )
        except FileNotFoundError:
            return f"\"{file_path}\" could not be found"
        except:
            return f"File \"{file_path}\" could not be decoded"
            

        t = time.time()
        match = self._recognize(*channels_samples)
        t = time.time() - t

        if match:
            match["match_time"] = t

        return match

    def recognize(self, filename):
        return self.recognize_file(filename)


class MicrophoneRecognizer(BaseRecognizer):
    default_chunksize = 8192
    default_format = pyaudio.paInt16
    default_channels = 2
    default_samplerate = 44100

    def __init__(self, audalign):
        super(MicrophoneRecognizer, self).__init__(audalign)
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.data = []
        self.channels = MicrophoneRecognizer.default_channels
        self.chunksize = MicrophoneRecognizer.default_chunksize
        self.samplerate = MicrophoneRecognizer.default_samplerate
        self.recorded = False

    def start_recording(
        self,
        channels=default_channels,
        samplerate=default_samplerate,
        chunksize=default_chunksize,
    ):
        print("* start recording")
        self.chunksize = chunksize
        self.channels = channels
        self.recorded = False
        self.samplerate = samplerate

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        self.stream = self.audio.open(
            format=self.default_format,
            channels=channels,
            rate=samplerate,
            input=True,
            frames_per_buffer=chunksize,
        )

        self.data = [[] for i in range(channels)]

    def process_recording(self):
        print("* recording")
        data = self.stream.read(self.chunksize)
        nums = np.fromstring(data, np.int16)
        # print(nums)
        for c in range(self.channels):
            self.data[c].extend(nums[c :: self.channels])

    def stop_recording(self):
        print("* done recording")
        self.stream.stop_stream()
        self.stream.close()
        self.stream = None
        self.recorded = True

    def recognize_recording(self):
        if not self.recorded:
            raise NoRecordingError("Recording was not complete/begun")
        return self._recognize(*self.data)

    def get_recorded_time(self):
        return len(self.data[0]) / self.samplerate

    def recognize(self, seconds=10):
        self.start_recording()
        for i in range(0, int(self.samplerate / self.chunksize * int(seconds))):
            self.process_recording()
        self.stop_recording()
        return self.recognize_recording()


class NoRecordingError(Exception):
    pass
