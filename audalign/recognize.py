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

    def recognize(self, file_path, filter_matches):
        try:
            channel_samples, self.Fs = decoder.read(
                file_path, limit=self.audalign.limit
            )
        except FileNotFoundError:
            return f'"{file_path}" not found'
        except:
            return f'File "{file_path}" could not be decoded'

        file_name = os.path.basename(file_path)

        t = time.time()
        matches = self.find_matches(channel_samples, file_name, Fs=self.Fs)
        rough_match = self.align_matches(matches)

        file_match = None
        if len(rough_match) > 0:
            file_match = self.process_results(rough_match, filter_matches)
        t = time.time() - t

        result = {}

        if file_match:
            result["match_time"] = t
            result["match_info"] = file_match
            return result

        return None

    def find_matches(self, samples, file_name, Fs=fingerprint.DEFAULT_FS):
        print(f'Fingerprinting "{file_name}"')
        target_mapper = fingerprint.fingerprint(samples, Fs=Fs)
        matches = []

        print(f"Finding Matches...  ", end="")
        for audio_file in self.audalign.fingerprinted_files:
            if audio_file[0].lower() != file_name.lower():
                already_hashes = audio_file[1]
                for t_hash in target_mapper.keys():
                    if t_hash in already_hashes.keys():
                        for t_offset in target_mapper[t_hash]:
                            for a_offset in already_hashes[t_hash]:
                                sample_difference = a_offset - t_offset
                                matches.append([audio_file[0], sample_difference])
        return matches

    def align_matches(self, matches):
        """
            Finds hash matches that align in time with other matches and finds
            consensus about which hashes are "true" signal from the audio.

            Returns a dictionary with match information.
        """
        print("Aligning matches")
        # align by sample_differences
        sample_difference_counter = {}
        for file_name, sample_difference in matches:
            if file_name not in sample_difference_counter:
                sample_difference_counter[file_name] = {}
            if sample_difference not in sample_difference_counter[file_name]:
                sample_difference_counter[file_name][sample_difference] = 0
            sample_difference_counter[file_name][sample_difference] += 1

        return sample_difference_counter

    def process_results(self, results, filter_matches=1):

        complete_match_info = {}

        for file_name in results:
            match_offsets = []
            offset_count = []
            offset_diff = []
            for sample_difference, num_of_matches in results[file_name].items():
                match_offsets.append((num_of_matches, sample_difference))
            match_offsets = sorted(match_offsets, reverse=True, key=lambda x: x[0])
            if match_offsets[0][0] <= filter_matches:
                continue
            for i in match_offsets:
                if i[0] <= filter_matches:
                    continue
                offset_count.append(i[0])
                offset_diff.append(i[1])

            complete_match_info[file_name] = {}
            complete_match_info[file_name][self.audalign.CONFIDENCE] = offset_count
            complete_match_info[file_name][self.audalign.OFFSET_SAMPLES] = offset_diff

            # extract idenfication

            complete_match_info[file_name][self.audalign.OFFSET_SECS] = []
            for i in offset_diff:
                nseconds = round(
                    float(i)
                    / fingerprint.DEFAULT_FS
                    * fingerprint.DEFAULT_WINDOW_SIZE
                    * fingerprint.DEFAULT_OVERLAP_RATIO,
                    5,
                )
                complete_match_info[file_name][self.audalign.OFFSET_SECS].append(
                    nseconds
                )

        if len(complete_match_info) == 0:
            return self.process_results(results, filter_matches=0)

        return complete_match_info
