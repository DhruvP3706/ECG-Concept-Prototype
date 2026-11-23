#!/bin/bash
# Experiment B scratch baseline, inner folds 1-3, sequential, single chain. (An earlier launch of this script was interrupted by a session restart; its two partial fold-1 run dirs are aborted and not used.)
cd /c/Users/lenovo/Desktop/ECG-Concept-Prototype
rm -f results/phase1_runs/B_folds123_done.flag
( while true; do nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader >> results/phase1_runs/B_folds123_gpu_v2.csv; sleep 15; done ) &
GPUPID=$!
for k in 1 2 3; do
  L=results/phase1_runs/B_fold${k}_stdout_v2.log
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -I scripts/train_v3.py --config experiments/phase1/B_ptbxl_scratch.json --inner-fold $k > $L 2>&1
  echo "EXIT $?" >> $L
done
kill $GPUPID
echo ALLDONE > results/phase1_runs/B_folds123_done.flag
