lr=0.001
wd=0.0
balanced=true
optimizer=sgd
auxiliary_head=true
sbatch -J "full-setr-cbis-multi-lr${lr}_wd${wd}" slurm/mmsegmentation/setr_cbis.sbatch ${lr} ${wd} full "cbis-multi" ${balanced} ${optimizer} ${auxiliary_head}