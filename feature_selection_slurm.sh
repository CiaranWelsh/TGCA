#!/bin/sh
conda activate py36

wd="/mnt/nfs/home/b7053098/ciaran/TGCA"
echo $wd
echo ${wd}/tgca/ga_for_cluster.py
for num_features in {1..50}; do
  for num_clusters in {3..25}; do
    echo "${num_features}_${num_clusters}"
    FLAGS="--ntasks=1 --cpus-per-task=1 --job-name=fs_${num_features}_${num_clusters}"
    sbatch $FLAGS --wrap="python ${wd}/tgca/ga_for_cluster.py $num_features $num_clusters"
  done
done
