lr=0.001
wd=0.0
sbatch -J "full-setr-cbis-multi-lr${lr}_wd${wd}" slurm/mmsegmentation/full_setr_cbis-multi.sbatch ${lr} ${wd}