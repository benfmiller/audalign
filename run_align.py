#! /usr/bin/env python3
import json
import numpy as np
import argparse

"""A script to run an audalign alignment"""

parser = argparse.ArgumentParser(description="Niftier running.")
parser.add_argument(
    "-f",
    "--files",
    type=str,
    help="files to fingerprint",
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
    import pprint
    import pickle
    import os
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

    ada = ad.Audalign(
        accuracy=args.accuracy,
        hash_style=args.hash_style,
        freq_threshold=args.threshold,
        num_processors=args.num_processors,
        multiprocessing=multiprocessing,
    )
    t = time.time()

    try:
        print()
        results = ada.align(
            args.files,
            technique=args.technique,
            cor_sample_rate=args.sample_rate,
            locality=args.locality,
            volume_threshold=args.volume_threshold,
            img_width=args.img_width,
            destination_path=args.destination,
            write_extension=args.write_extention,
        )
        if args.fine_align:
            results = ada.fine_align(
                results,
                technique=args.fine_technique,
                locality=args.fine_locality,
                cor_sample_rate=args.fine_sample_rate,
                img_width=args.fine_img_width,
                volume_threshold=args.fine_volume_threshold,
                write_extension=args.write_extention,
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
    ada.pretty_print_results(results)

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
