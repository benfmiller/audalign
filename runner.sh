#!/usr/bin/bash

write_file="workj_results.txt"

if test -f "workj_results.txt"; then
    cp $write_file workj_results_last.txt
    > $write_file
    echo "last run cleared, new tests"
fi

echo "Starting run" >> $write_file
echo "------------" >> $write_file
num_runs=0

num_processors=6
technique='fingerprints'
sample_rate=44100
files='audio_files/shifts/'
img_width=0.5
volume_threshold=215

files='audio_files/audio_sync/20200602/'
files='audio_files/audio_sync/20201204/'
files='audio_files/audio_sync/segs/'
files='audio_files/audio_sync/time_diff_claps/line/'
files='audio_files/audio_sync/time_diff_claps/rec/'
files='audio_files/audio_sync/trec1/'
files='audio_files/audio_sync/trec2/'
files='audio_files/audio_sync/trecs/'

# gotta test the noise reduced files, too

# this works great but I don't like the arbitrary 999999
# python3 figuring.py -a 1 | tac | grep -F -m1 -B 999999 '}' | head -n -1 | tac



accuracy=( 1 2 3 4 )
hash_style=( "panako_mod" "panako" "base" "base_three" )
freq_threshold=(0 50 100 200)
locality=(0 5 10 20 40)
for f in "${freq_threshold[@]}"; do
    for h in "${hash_style[@]}"; do 
        for a in "${accuracy[@]}"; do
        ((num_runs=num_runs+1))
        echo >> $write_file
        echo "    run number $num_runs------------------------" >> $write_file
        echo "    Hash Style '$h', Accuracy '$a', Frequency Threshold '$f'">> $write_file
        echo >> $write_file

        # python3 figuring.py -a 1 | tac | sed '/}/Q' | tac | grep "\S" >> workj_results.txt
        # python3 figuring.py -a 1 | tac | sed '/}/Q' | tac >> workj_results.txt
        done
    done
done
# for l in "${locality[@]}"; do 
# # only panako_mod, accuracy 2-4,
# done
echo "$(date +"%T"): Running number $num_runs"
# python3 figuring.py -a 1 | tac | sed '/}/Q' | tac | grep "\S" >> workj_results.txt
# let time_took = $results | tail -n 1
# echo $now
# echo "$time_took"

# temp_results=$(< workj_temp.txt)
# temp_results=""
# while IFS= read -r line; do
#     temp_results=$temp_results$line

    # echo "Text read from file: $line"
# done < workj_temp.txt
# $temp_results | grep "took" | cat
# echo $temp_results
# python3 figuring.py -a 1 > workj_temp.txt
# echo $img_width

echo >> $write_file
echo "--------------------" >> $write_file
echo "$num_runs total runs" >> $write_file
echo
echo "--------------------"
echo "$num_runs total runs"
echo
echo "All Done!!"