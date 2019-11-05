#!/bin/sh
conda activate py36

wd=${dirname $1}

for num_features in {1..10}; do
  for num_clusters in {3..10}; do
    echo "${num_features}_${num_clusters}"
    FLAGS="--ntasks=1 --cpus-per-task=1 --job-name=fs_${num_features}_${num_clusters}"
    sbatch $FLAGS --wrap="$wd/tgca/ga_for_cluster.py $num_features $num_clusters"
  done
done
