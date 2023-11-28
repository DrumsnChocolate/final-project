
lr=0.1
wd=0.001
sbatch -J "vpt-setr-cbis-multi-lr${lr}_wd${wd}" slurm/mmsegmentation/vpt_setr_cbis-multi.sbatch ${lr} ${wd}
