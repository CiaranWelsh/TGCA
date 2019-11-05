#!/bin/sh

for num_features in {1..20}; do
    for num_clusters in {3..20}; do
        FLAGS="--ntasks=1 --cpus-per-task=1 --job-name=fs_${num_features}_${num_clusters}"
        sbatch $FLAGS tgca/ga_for_cluster.py $num_features $num_clusters
    done
done
