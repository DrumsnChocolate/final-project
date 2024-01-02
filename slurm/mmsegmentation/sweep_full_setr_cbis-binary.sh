wd=0.001
balanced=true
optimizer=sgd
for lr in 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001; do
  sbatch -J "full-setr-cbis-binary-lr${lr}_wd${wd}" slurm/mmsegmentation/setr_cbis.sbatch ${lr} ${wd} full "cbis-binary" ${balanced} ${optimizer}
done