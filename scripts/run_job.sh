#!/bin/sh
#OAR -l nodes=1, walltime=2:00:00
#OAR -O /home/alecoutre/runs/run_%jobid%.output
#OAR -E /home/alecoutre/runs/run_%jobid%.error
set -xv
#OAR -p cluster='chifflet'

file=$1

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/home/alecoutre/cuda/lib64

while read a b c d e f
do
    python3 ../python/run.py $a $b $c $d $e $f
done < $file
