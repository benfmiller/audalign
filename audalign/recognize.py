# encoding: utf-8
import audalign.fingerprint as fingerprint
import audalign.filehandler as filehandler
import numpy as np
import time
import os


class FileRecognizer:
    def __init__(self, audalign):
        """
        Creates FileRecognizer object

        Parameters
        ----------
        audalign : Audalign
            instance of Audalign
        
        Returns
        -------
        None
        """
        self.audalign = audalign
        self.Fs = fingerprint.DEFAULT_FS

    def recognize(self, file_path, filter_matches):
        """
        Recognizes given file against already fingerprinted files

        Parameters
        ----------
        file_path : str
            file path of target file
        filter_matches : int
            only returns information on match counts greater than filter_matches
        
        Returns
        -------
        match_result : dict
            dictionary containing match time and match info
            
            or

            None : if no match
        """

        t = time.time()
        matches = self.find_matches(file_path, Fs=self.Fs)
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

    def find_matches(self, file_path, Fs=fingerprint.DEFAULT_FS):
        """
        fingerprints target file, then finds every occurence of exact same hashes in already
        fingerprinted files

        Parameters
        ----------
        samples : array of decoded file
            array of decoded file from filehandler.read
        file_name : str
            base name of target file

        Returns
        -------
        Matches: list[str, int]
            list of all matches, file_name match and corresponding offset
        """
        file_name = os.path.basename(file_path)

        if file_name not in self.audalign.file_names:
            try:
                samples, self.Fs = filehandler.read(file_path)
                print(f'Fingerprinting "{file_name}"')
                target_mapper = fingerprint.fingerprint(samples, Fs=Fs)
            except FileNotFoundError:
                print(f'"{file_path}" not found')
                return {}
            except:
                print(f'File "{file_path}" could not be decoded')
                return {}
        else:
            for audio_file in self.audalign.fingerprinted_files:
                if audio_file[0] == file_name:
                    target_mapper = audio_file[1]
                    break

        matches = []

        print(f"{file_name}: Finding Matches...  ", end="")
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
        takes matches from find_matches and converts it to a dictionary of counts per offset and file name

        Parameters
        ----------
        matches : list[str, int]
            list of matches from find_matches

        Returns
        -------
        sample_difference_counter : dict{str{int}}
            of the form dict{file_name{number of matching offsets}}
        """

        print("Aligning matches")

        sample_difference_counter = {}
        for file_name, sample_difference in matches:
            if file_name not in sample_difference_counter:
                sample_difference_counter[file_name] = {}
            if sample_difference not in sample_difference_counter[file_name]:
                sample_difference_counter[file_name][sample_difference] = 0
            sample_difference_counter[file_name][sample_difference] += 1

        return sample_difference_counter

    def process_results(self, results, filter_matches=1):
        """
        Takes matches from align_matches, filters and orders them, returns dictionary of match info

        Parameters
        ----------
        results : dict{str{int}}
            of the form dict{file_name{number of matching offsets}}
        
        filter_matches : int
            cutout all matches equal to or less than in frequency, goes down if no matches found above filter
        
        Returns
        -------
        match_info : dict{dict{}}
            dict of file_names with match info as values
        """

        complete_match_info = {}

        for file_name in results.keys():
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
            return self.process_results(results, filter_matches=filter_matches - 1)

        return complete_match_info
