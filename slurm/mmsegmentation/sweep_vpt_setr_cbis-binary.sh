# what's the effect of different dataset sizes on performance?
# what's the effect of different numbers of iterations on performance?
# what's the effect of different image sizes on performance?

# let's first investigate the optimal lr and weight decay and prompt dropout, at img size 512 and iteration count 160k
#for lr in 0.01 0.005 0.001 0.0005 0.0001; do
#  for wd in 0.01 0.001 0.0001 0.0; do
#    sbatch -J "vpt-setr-cbis-binary-lr${lr}_wd${wd}" slurm/mmsegmentation/setr_cbis.sbatch ${lr} ${wd} vpt "cbis-binary"
#  done
#done
#lr=0.1
#wd=0.001
#sbatch -J "vpt-setr-cbis-binary-lr${lr}_wd${wd}" slurm/mmsegmentation/setr_cbis.sbatch ${lr} ${wd} vpt "cbis-binary"

wd=0.001
for lr in 0.05 0.01 0.005 0.001 0.0005 0.0001; do
  sbatch -J "vpt-setr-cbis-binary-lr${lr}_wd${wd}" slurm/mmsegmentation/setr_cbis.sbatch ${lr} ${wd} vpt "cbis-binary"
done

#for dataset_size in 3103 2500 2000 1500 1000 500; do
#  for iterations in 160000 80000 40000; do
#    for img_size in 1000 800 600 400; do
#      for prompt_dropout in 0.1 0.0; do
#        for lr in 0.01 0.005 0.001 0.0005 0.0001; do
#          for wd in 0.01 0.001 0.0001 0.0; do
#            echo "todo: pass these arguments to the sbatch script and feed them as the right config options"
#          done
#        done
#      done
#    done
#  done
#done