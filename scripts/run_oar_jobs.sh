#!/bin/sh
#OAR -l nodes=1,walltime=14:00:00
#OAR --array-param-file /home/alecoutre/rasta/python/scripts/configs/conf_29_06.txt
#OAR -O /home/alecoutre/runs/run_%jobid%.output
#OAR -E /home/alecoutre/runs/run_%jobid%.error
#OAR -p cluster='chifflet'

echo "parametres: "$*

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:~/cuda/lib64

python3 ../run.py -m=$1 -b=$2 -e=$3 -f=$4 -n=$5 -d=$6 --train_path=$7 --val_path=$8

