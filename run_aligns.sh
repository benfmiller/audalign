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

# expecting 256 matches for fingerprints
files=( 'audio_files/shifts/' 'audio_files/audio_sync/20200602/' )
for file in "${files[@]}"; do
    accuracy=( 1 2 3 4 )
    hash_style=( "panako_mod" "panako" "base" "base_three" )
    freq_threshold=(0 25 50 100 200)
    for f in "${freq_threshold[@]}"; do
        for h in "${hash_style[@]}"; do 
            for a in "${accuracy[@]}"; do
            ((num_runs=num_runs+1))
            echo >> $write_file
            echo "    run number $num_runs------------------------" >> $write_file
            echo "    '$file', Hash Style '$h', Accuracy '$a', Frequency Threshold '$f'">> $write_file
            echo >> $write_file
            echo "$(date +"%T"): Running number $num_runs"
            echo "    '$file', Hash Style '$h', Accuracy '$a', Frequency Threshold '$f'"

            # result=$(python3 figuring.py -a $a -s $h -r $f -f $file| tac | sed '/}/Q' | tac | grep "\S")
            result=$(python3 figuring.py -a $a -s $h -r $f -f $file)
            echo "$result" | rg "Total fingerprints" >> $write_file
            echo >> $write_file
            echo "$result" | tac | sed '/}/Q' | tac | rg -N "\S" >> $write_file

            done
        done
    done

    locality=( 0 5 15 )
    accuracy=( 2 3 )
    hash_style=( "panako_mod" "base" )
    freq_threshold=(0 50 100 )
    for l in "${locality[@]}"; do 
        for f in "${freq_threshold[@]}"; do
            for h in "${hash_style[@]}"; do 
                for a in "${accuracy[@]}"; do
                ((num_runs=num_runs+1))
                echo >> $write_file
                echo "    run number $num_runs------------------------" >> $write_file
                echo "    '$file', Hash Style '$h', Accuracy '$a', Frequency Threshold '$f', Locality '$l'">> $write_file
                echo >> $write_file
                echo "$(date +"%T"): Running number $num_runs"
                echo "    '$file', Hash Style '$h', Accuracy '$a', Frequency Threshold '$f', Locality '$l'"

                # python3 figuring.py -a $a -s $h -r $f -l $l -f $file| tac | sed '/}/Q' | tac | grep "\S" >> $write_file
                result=$(python3 figuring.py -a $a -s $h -r $f -l $l -f $file)
                echo "$result" | rg "Total fingerprints" >> $write_file
                echo >> $write_file
                echo "$result" | tac | sed '/}/Q' | tac | rg -N "\S" >> $write_file

                done
            done
        done
    done
done
files=( 'audio_files/shifts/' 'audio_files/audio_sync/20200602/' 'audio_files/audio_sync/20201204/' 'audio_files/audio_sync/trec1/' )
for file in "${files[@]}"; do
    accuracy=( 4 )
    hash_style=( "panako_mod")
    freq_threshold=( 50 100 )
    for f in "${freq_threshold[@]}"; do
        for h in "${hash_style[@]}"; do 
            for a in "${accuracy[@]}"; do
            ((num_runs=num_runs+1))
            echo >> $write_file
            echo "    run number $num_runs------------------------" >> $write_file
            echo "    '$file', Hash Style '$h', Accuracy '$a', Frequency Threshold '$f'">> $write_file
            echo >> $write_file
            echo "$(date +"%T"): Running number $num_runs"
            echo "    '$file', Hash Style '$h', Accuracy '$a', Frequency Threshold '$f'"

            # python3 figuring.py -a $a -s $h -r $f -f $file| tac | sed '/}/Q' | tac | grep "\S" >> $write_file
            result=$(python3 figuring.py -a $a -s $h -r $f -f $file)
            echo "$result" | rg "Total fingerprints" >> $write_file
            echo >> $write_file
            echo "$result" | tac | sed '/}/Q' | tac | rg -N "\S" >> $write_file

            done
        done
    done

    locality=( 5 15 )
    accuracy=( 4 )
    hash_style=( "panako_mod" )
    freq_threshold=( 50 100 )
    for l in "${locality[@]}"; do 
        for f in "${freq_threshold[@]}"; do
            for h in "${hash_style[@]}"; do 
                for a in "${accuracy[@]}"; do
                ((num_runs=num_runs+1))
                echo >> $write_file
                echo "    run number $num_runs------------------------" >> $write_file
                echo "    '$file', Hash Style '$h', Accuracy '$a', Frequency Threshold '$f', Locality '$l'">> $write_file
                echo >> $write_file
                echo "$(date +"%T"): Running number $num_runs"
                echo "    '$file', Hash Style '$h', Accuracy '$a', Frequency Threshold '$f', Locality '$l'"

                # python3 figuring.py -a $a -s $h -r $f -l $l -f $file| tac | sed '/}/Q' | tac | grep "\S" >> $write_file
                result=$(python3 figuring.py -a $a -s $h -r $f -l $l -f $file)
                echo "$result" | rg "Total fingerprints" >> $write_file
                echo >> $write_file
                echo "$result" | tac | sed '/}/Q' | tac | rg -N "\S" >> $write_file

                done
            done
        done
    done
done

# ------------------------------- correlation

write_file="workj_results_corelation.txt"

if test -f $write_file; then
    cp $write_file workj_results_last_correlation.txt
    > $write_file
    echo "last run cleared, new tests"
fi

technique="correlation"
files=( 'audio_files/shifts/' 'audio_files/audio_sync/20200602/' 'audio_files/audio_sync/20201204/' 'audio_files/audio_sync/trec1/' )
for file in "${files[@]}"; do # TODO make sure all files are run, not quite right
    freq_threshold=( 25 50 100 )
    sample_rates=( 4000 8000 16000 44100 )
    for r in "${freq_threshold[@]}"; do
        for m in "${sample_rates[@]}"; do 
            ((num_runs=num_runs+1))
            echo >> $write_file
            echo "    run number $num_runs------------------------" >> $write_file
            echo "    '$file', Frequency Threshold '$r', Sample Rate '$m', $technique">> $write_file
            echo >> $write_file
            echo "$(date +"%T"): Running number $num_runs"
            echo "    '$file', Frequency Threshold '$r', Sample Rate '$m', $technique"

            result=$(python3 figuring.py -t $technique -r $r -m $m -f $file)
            echo "$result" | rg "Total fingerprints" >> $write_file
            echo >> $write_file
            echo "$result" | tac | sed '/}/Q' | tac | rg -N "\S" >> $write_file
            done
        done
    done

    locality=( 5 15 25 )
    freq_threshold=( 50 100 )
    sample_rates=( 8000 16000 44100 )
    for l in "${locality[@]}"; do 
        for r in "${freq_threshold[@]}"; do
            for m in "${sample_rates[@]}"; do 
                ((num_runs=num_runs+1))
                echo >> $write_file
                echo "    run number $num_runs------------------------" >> $write_file
                echo "    '$file', Frequency Threshold '$r', Sample Rate '$m', Locality '$l', $technique">> $write_file
                echo >> $write_file
                echo "$(date +"%T"): Running number $num_runs"
                echo "    '$file', Frequency Threshold '$r', Sample Rate '$m', Locality '$l', $technique"

                result=$(python3 figuring.py -t $technique -r $r -m $m -l $l -f $file)
                echo "$result" | rg "Total fingerprints" >> $write_file
                echo >> $write_file
                echo "$result" | tac | sed '/}/Q' | tac | rg -N "\S" >> $write_file
                done
            done
        done
    done
done

# ------------------------------- correlation spectrogram

write_file="workj_results_corelation_spectrogram.txt"

if test -f $write_file; then
    cp $write_file workj_results_last_correlation_spectrogram.txt
    > $write_file
    echo "last run cleared, new tests"
fi

technique="correlation_spectrogram"
files=( 'audio_files/shifts/' 'audio_files/audio_sync/20200602/' 'audio_files/audio_sync/20201204/' 'audio_files/audio_sync/trec1/' )
for file in "${files[@]}"; do
    freq_threshold=( 25 50 100 )
    sample_rates=( 4000 8000 16000 44100 )
    for r in "${freq_threshold[@]}"; do
        for m in "${sample_rates[@]}"; do 
            ((num_runs=num_runs+1))
            echo >> $write_file
            echo "    run number $num_runs------------------------" >> $write_file
            echo "    '$file', Frequency Threshold '$r', Sample Rate '$m', $technique">> $write_file
            echo >> $write_file
            echo "$(date +"%T"): Running number $num_runs"
            echo "    '$file', Frequency Threshold '$r', Sample Rate '$m', $technique"

            result=$(python3 figuring.py -t $technique -r $r -m $m -f $file)
            echo "$result" | rg "Total fingerprints" >> $write_file
            echo >> $write_file
            echo "$result" | tac | sed '/}/Q' | tac | rg -N "\S" >> $write_file
            done
        done
    done

    locality=( 5 15 25 )
    freq_threshold=( 50 100 )
    sample_rates=( 8000 16000 44100 )
    for l in "${locality[@]}"; do 
        for r in "${freq_threshold[@]}"; do
            for m in "${sample_rates[@]}"; do 
                ((num_runs=num_runs+1))
                echo >> $write_file
                echo "    run number $num_runs------------------------" >> $write_file
                echo "    '$file', Frequency Threshold '$r', Sample Rate '$m', Locality '$l', $technique">> $write_file
                echo >> $write_file
                echo "$(date +"%T"): Running number $num_runs"
                echo "    '$file', Frequency Threshold '$r', Sample Rate '$m', Locality '$l', $technique"

                result=$(python3 figuring.py -t $technique -r $r -m $m -l $l -f $file)
                echo "$result" | rg "Total fingerprints" >> $write_file
                echo >> $write_file
                echo "$result" | tac | sed '/}/Q' | tac | rg -N "\S" >> $write_file
                done
            done
        done
    done
done
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