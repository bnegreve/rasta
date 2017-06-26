#!/bin/sh
#OAR -l nodes=1,walltime=09:00:00
#OAR --array-param-file /home/alecoutre/rasta/python/scripts/configs/multi_job.txt
#OAR -O /home/alecoutre/runs/run_%jobid%.output
#OAR -E /home/alecoutre/runs/run_%jobid%.error
#OAR -p cluster='chifflet'

echo "parametres: "$*

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/home/alecoutre/cuda/lib64

python3 ../run.py $*

