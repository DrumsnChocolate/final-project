lr=0.001
wd=0.0
sbatch -J "full-setr-cbis-binary-lr${lr}_wd${wd}" slurm/mmsegmentation/full_setr_cbis-binary.sbatch ${lr} ${wd}