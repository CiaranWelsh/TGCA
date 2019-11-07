#!/bin/bash
cluster=false
conda activate py36

# shellcheck disable=SC1073
if [ "$cluster" == true ] ; then
  wd="/mnt/nfs/home/b7053098/ciaran/TGCA"
else
  wd="/media/ncw135/DATA/TGCA"
fi

echo $wd
echo ${wd}/tgca/ga_for_cluster.py
for num_features in {3..25}; do
  for num_clusters in {2..20}; do
    command="python ${wd}/tgca/ga_for_cluster.py plotter --num_clusters $num_clusters --num_features $num_features --pca"
    if [ "$cluster" == true ] ; then
      FLAGS="--ntasks=1 --cpus-per-task=1 --job-name=fs_${num_features}_${num_clusters}"
      sbatch $FLAGS --wrap="$command"
    else
      $command
    fi
  done
done
