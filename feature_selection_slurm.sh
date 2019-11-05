#!/bin/bash
#SBATCH --job-name=feature_selection01
#SBATCH --output=cv_analysis_eis-%j-%a.out
#SBATCH --error=cv_analysis_eis-%j-%a.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6


#!/bin/bash
FLAGS="--ntasks=1 --cpus-per-task=1"
for num_features in {1..20}; do
    for num_clusters in {3..20}; do
        sbatch python ga_for_cluster.py $num_features $num_clusters
    done
done
