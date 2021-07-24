#!/usr/bin/bash
if test -f "workj_results.txt"; then
    cp workj_results.txt workj_results_last.txt
    > workj_results.txt
    echo "last run cleared, new tests"
fi

echo "Starting run" >> workj_results.txt
echo "------------" >> workj_results.txt
let num_runs=0

let accuracy=1
let hash_style="panako_mod"
let freq_threshold=100
let num_processors=6
let technique="fingerprints"
let locality=0
let sample_rate=44100
let files="audio_files/shifts/"
let img_width=0.5
let volume_threshold=215

let files="audio_files/audio_sync/20200602/"
let files="audio_files/audio_sync/20201204/"
let files="audio_files/audio_sync/segs/"
let files="audio_files/audio_sync/time_diff_claps/line/"
let files="audio_files/audio_sync/time_diff_claps/rec/"
let files="audio_files/audio_sync/trec1/"
let files="audio_files/audio_sync/trec2/"
let files="audio_files/audio_sync/trecs/"

# gotta test the noise reduced files, too

# this works great but I don't like the arbitrary 999999
# python3 figuring.py -a 1 | tac | grep -F -m1 -B 999999 '}' | head -n -1 | tac

((num_runs=num_runs+1))
echo >> workj_results.txt
echo "run number $num_runs------------------------" >> workj_results.txt


let accuracy=1
echo "Run number $num_runs"
# python3 figuring.py -a 1 | tac | sed '/}/Q' | tac >> workj_results.txt

echo >> workj_results.txt
echo "--------------------" >> workj_results.txt
echo "$num_runs total runs" >> workj_results.txt
echo
echo "--------------------"
echo "$num_runs total runs"
echo
echo "All Done!!"