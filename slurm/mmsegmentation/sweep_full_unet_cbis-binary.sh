wd=0.0005
lr=0.001
sbatch -J "full-unet-cbis-binary-lr${lr}_wd${wd}" slurm/mmsegmentation/unet_cbis.sbatch ${lr} ${wd}
#for lr in 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001; do
#  sbatch -J "full-unet-cbis-binary-lr${lr}_wd${wd}" slurm/mmsegmentation/unet_cbis.sbatch ${lr} ${wd}
#done