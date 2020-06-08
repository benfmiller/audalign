# encoding: utf-8
import audalign.fingerprint as fingerprint
import audalign.decoder as decoder
import numpy as np
import time


class BaseRecognizer(object):
    def __init__(self, audalign):
        self.audalign = audalign
        self.Fs = fingerprint.DEFAULT_FS

    def _recognize(self, channel_samples):
        matches = self.audalign.find_matches(channel_samples, Fs=self.Fs)
        return self.audalign.align_matches(matches)

    def recognize(self):
        pass  # base class does nothing


class FileRecognizer(BaseRecognizer):
    def __init__(self, audalign):
       super().__init__(audalign)

    def recognize_file(self, file_path):
        try:
            channel_samples, self.Fs, file_hash = decoder.read(
                file_path, limit=self.audalign.limit
            )
        except FileNotFoundError:
            return f'"{file_path}" not found'
        except:
            return f'File "{file_path}" could not be decoded'

        t = time.time()
        match = self._recognize(channel_samples)
        t = time.time() - t

        if match:
            match["match_time"] = t

        return match

    def recognize(self, filename):
        return self.recognize_file(filename)
