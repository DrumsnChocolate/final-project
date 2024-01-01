lr=0.1
wd=0.001
balanced=true
optimizer=sgd
sbatch -J "vpt-setr-cbis-multi-lr${lr}_wd${wd}" slurm/mmsegmentation/setr_cbis.sbatch ${lr} ${wd} vpt "cbis-multi" ${balanced} ${optimizer}