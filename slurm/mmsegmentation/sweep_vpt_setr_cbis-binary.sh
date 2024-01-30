# what's the effect of different dataset sizes on performance?
# what's the effect of different numbers of iterations on performance?
# what's the effect of different image sizes on performance?


wd=0.001
balanced=false
optimizer=sgd
auxiliary_head=true
for lr in 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001; do
  sbatch -J "vpt-setr-cbis-binary-lr${lr}_wd${wd}" slurm/mmsegmentation/setr_cbis.sbatch ${lr} ${wd} vpt "cbis-binary" ${balanced} ${optimizer} ${auxiliary_head}
done