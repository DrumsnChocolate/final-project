lr=0.001
wd=0.0
sbatch -J "full-setr-cbis-multi-lr${lr}_wd${wd}" slurm/mmsegmentation/setr_cbis.sbatch ${lr} ${wd} full "cbis-multi"