#!/bin/bash
time_list="short_time_range_file.txt"
while IFS= read -r t
do
    echo $t
    sbatch job.sbatch $t
done < $time_list
#while IFS= read -r st
#do
#    sbatch flu_hospjob.sbatch $st $input_hrzn
#            #qreg
#done < $st_list

#while IFS= read -r st
#do
#    while IFS= read -r cnty
#    do
#        echo $st $cnty
#        sbatch job.sbatch $st $cnty $input_hrzn
#    done < "$st"
#done < "$st_list"
