# encoding: utf-8
import audalign.fingerprint as fingerprint
import audalign.decoder as decoder
import numpy as np
import time
import os


class FileRecognizer:
    def __init__(self, audalign):
        self.audalign = audalign
        self.Fs = fingerprint.DEFAULT_FS

    def recognize(self, file_path):
        try:
            channel_samples, self.Fs, file_hash = decoder.read(
                file_path, limit=self.audalign.limit
            )
        except FileNotFoundError:
            return f'"{file_path}" not found'
        except:
            return f'File "{file_path}" could not be decoded'

        file_name, extension = os.path.splitext(os.path.basename(file_path))
        file_name += extension

        t = time.time()
        matches = self.find_matches(channel_samples, file_name, Fs=self.Fs)
        file_match = self.align_matches(matches)
        t = time.time() - t

        if file_match:
            file_match["match_time"] = t

        return file_match

    def get_file_id(self, name):
        for i in self.audalign.fingerprinted_files:
            if i[0] == name:
                return i[2]

    def find_matches(self, samples, file_name, Fs=fingerprint.DEFAULT_FS):
        target_mapper = fingerprint.fingerprint(samples, Fs=Fs)
        matches = []

        for audio_file in self.audalign.fingerprinted_files:
            if audio_file[0].lower() != file_name.lower():
                already_hashes = audio_file[1]
                for t_hash in target_mapper.keys():
                    if t_hash in already_hashes.keys():
                        for t_offset in target_mapper[t_hash]:
                            for a_offset in already_hashes[t_hash]:
                                diff = a_offset - t_offset
                                matches.append([audio_file[0], diff])
        return matches

    def align_matches(self, matches):
        """
            Finds hash matches that align in time with other matches and finds
            consensus about which hashes are "true" signal from the audio.

            Returns a dictionary with match information.
        """
        # align by diffs
        diff_counter = {}
        largest_match_offset = 0
        largest_match_count = 0
        file_name = -1
        for pair in matches:
            sid, diff = pair
            if diff not in diff_counter:
                diff_counter[diff] = {}
            if sid not in diff_counter[diff]:
                diff_counter[diff][sid] = 0
            diff_counter[diff][sid] += 1

            if diff_counter[diff][sid] > largest_match_count:
                largest_match_offset = diff
                largest_match_count = diff_counter[diff][sid]
                file_name = sid

        # extract idenfication
        file_id = self.get_file_id(file_name)

        # return match info
        nseconds = round(
            float(largest_match_offset)
            / fingerprint.DEFAULT_FS
            * fingerprint.DEFAULT_WINDOW_SIZE
            * fingerprint.DEFAULT_OVERLAP_RATIO,
            5,
        )
        audio_file = {
            self.audalign.FILE_ID: file_id,
            self.audalign.FILE_NAME: file_name,
            self.audalign.CONFIDENCE: largest_match_count,
            self.audalign.OFFSET_SAMPLES: int(largest_match_offset),
            self.audalign.OFFSET_SECS: nseconds,
        }
        return audio_file
