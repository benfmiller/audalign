#! /usr/bin/env python3
import argparse
import json

import numpy as np

"""A script to run an audalign alignment"""

parser = argparse.ArgumentParser(description="Niftier running.")
parser.add_argument(
    "-f",
    "--files",
    type=str,
    help="files to fingerprint",
    required=True,
)
parser.add_argument(
    "--fine-align", action="store_true", help="if present, runs a fine alignment"
)
parser.add_argument(
    "-d",
    "--destination",
    type=str,
    required=False,
    default=None,
)
parser.add_argument("-w", "--write-extention", type=str, required=False, default=None)
parser.add_argument(
    "--write-multi-channel",
    help="If present, only writes a multi-channel file as output",
    action="store_true",
)

parser.add_argument(
    "--write-multi-channel-fine",
    help="If present, only writes a multi-channel file as output for fine align",
    action="store_true",
)


parser.add_argument(
    "-t",
    "--technique",
    type=str,
    help="alignment technique",
    required=False,
    default="fingerprints",
)
parser.add_argument(
    "-l",
    "--locality",
    type=float,
    help="locality",
    required=False,
    default=None,
)
parser.add_argument(
    "-m",
    "--sample-rate",
    type=int,
    help="sample rate to read the file in",
    required=False,
    default=44100,
)
parser.add_argument(
    "-a",
    "--accuracy",
    type=int,
    help="accuracy for fingerprints",
    required=False,
    default=2,
)
parser.add_argument(
    "-s",
    "--hash-style",
    type=str,
    help="hash style for fingerprints",
    required=False,
    default="panako_mod",
)
parser.add_argument(
    "-r",
    "--threshold",
    type=int,
    help="frequency threshold",
    required=False,
    default=100,
)
parser.add_argument(
    "-n",
    "--num-processors",
    type=int,
    help="number of processors to use",
    required=False,
    default=6,
)
parser.add_argument(
    "-i",
    "--img-width",
    type=float,
    help="image width for visual",
    required=False,
    default=0.5,
)
parser.add_argument(
    "-v",
    "--volume-threshold",
    type=float,
    help="volume threshold for visual",
    required=False,
    default=215,
)
parser.add_argument(
    "--write_results",
    action="store_true",
    help='if present, writes results to "last_results.json"',
)
parser.add_argument(
    "--fine-technique",
    type=str,
    help="fine alignment technique",
    required=False,
    default="correlation",
)
parser.add_argument(
    "--fine-locality",
    type=float,
    help="fine alignment locality",
    required=False,
    default=None,
)
parser.add_argument(
    "--fine-sample-rate",
    type=int,
    help="fine alignment sample rate to convert the files to",
    required=False,
    default=8000,
)
parser.add_argument(
    "--fine-img-width",
    type=float,
    help="fine alingment image width for visual",
    required=False,
    default=0.5,
)
parser.add_argument(
    "--fine-volume-threshold",
    type=float,
    help="fine alignment volume threshold for visual",
    required=False,
    default=215,
)

args = parser.parse_args()

# print(args.)
# print(args.sample_rate)
# print(args.files)
# print(args.threshold)
# print(args.locality)

# __name__ = "blah"


def main(args):
    import os
    import pickle
    import pprint
    import time

    import audalign as ad

    results = None
    results_rank = None

    multiprocessing = True
    if args.num_processors == 1:
        multiprocessing = False
    elif args.num_processors == 0:
        args.num_processors = None
    if args.locality == 0:
        args.locality = None

    if args.technique == "fingerprints":
        recognizer = ad.FingerprintRecognizer()
        recognizer.config.set_accuracy(args.accuracy)
        recognizer.config.set_hash_style(args.hash_style)
    elif args.technique == "correlation":
        recognizer = ad.CorrelationRecognizer()
    elif args.technique == "correlation_spectrogram":
        recognizer = ad.CorrelationSpectrogramRecognizer()
    elif args.technique == "visual":
        recognizer = ad.VisualRecognizer()
        recognizer.config.volume_threshold = args.volume_threshold
        recognizer.config.img_width = args.img_width
    else:
        raise ValueError(
            f"technique '{args.technique}' must be 'fingerprints', 'correlation', 'correlation_spectrogram', or 'visual'"
        )

    recognizer.config.freq_threshold = args.threshold
    recognizer.config.num_processors = args.num_processors
    recognizer.config.multiprocessing = multiprocessing
    recognizer.config.sample_rate = args.sample_rate
    recognizer.config.locality = args.locality

    t = time.time()

    try:
        print()
        results = ad.align(
            args.files,
            destination_path=args.destination,
            write_extension=args.write_extention,
            write_multi_channel=args.write_multi_channel,
            recognizer=recognizer,
        )
        if args.fine_align:
            if args.fine_technique == "fingerprints":
                recognizer = ad.FingerprintRecognizer()
                recognizer.config.set_accuracy(args.accuracy)
                recognizer.config.set_hash_style(args.hash_style)
            elif args.fine_technique == "correlation":
                recognizer = ad.CorrelationRecognizer()
            elif args.fine_technique == "correlation_spectrogram":
                recognizer = ad.CorrelationSpectrogramRecognizer()
            elif args.fine_technique == "visual":
                recognizer = ad.VisualRecognizer()
                recognizer.config.volume_threshold = args.fine_volume_threshold
                recognizer.config.img_width = args.fine_img_width
            else:
                raise ValueError(
                    f"fine_technique '{args.fine_technique}' must be 'fingerprints', 'correlation', 'correlation_spectrogram', or 'visual'"
                )
            recognizer.config.freq_threshold = args.threshold
            recognizer.config.num_processors = args.num_processors
            recognizer.config.multiprocessing = multiprocessing
            recognizer.config.sample_rate = args.sample_rate
            recognizer.config.locality = args.locality
            results = ad.fine_align(
                results,
                write_extension=args.write_extention,
                write_multi_channel=args.write_multi_channel_fine,
                destination_path=args.destination,
            )

        print()

        t = time.time() - t

    except KeyboardInterrupt:
        t = time.time() - t
        print(f"\nRan for {ad.seconds_to_min_hrs(t)}.")
        return

    if results is not None and args.write_results:
        with open("last_results.json", "w") as f:
            json.dump(results, f, cls=NpEncoder)

    # --------------------------------------------------------------------------
    ad.pretty_print_results(results)

    sum_, count, max_, min_ = 0, 0, 0, 9999999999
    if results is not None:

        for target in results["rankings"]["match_info"].keys():
            temp_results = results["rankings"]["match_info"][target]
            if temp_results == 0:
                max_ = max(max_, 0)
                min_ = min(min_, 0)
                sum_ += 0
                count += 1
                continue
            for rank in temp_results.values():
                max_ = max(max_, rank)
                min_ = min(min_, rank)
                sum_ += rank
                count += 1
        print()
        print(
            f"Rankings -- Count: {count}, Sum: {sum_}, Min: {min_}, Average: {sum_ / count}, Max: {max_}"
        )

    print()
    print(f"It took {ad.seconds_to_min_hrs(t)} seconds to complete.")


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


if __name__ == "__main__":
    main(args=args)
