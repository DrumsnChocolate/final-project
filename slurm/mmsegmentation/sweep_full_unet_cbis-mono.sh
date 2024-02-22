lr=0.0001
size=256
loss="cross_entropy"
sbatch -J "full-unet-cbis-mono-lr${lr}" slurm/mmsegmentation/unet_cbis.sbatch ${lr} ${size} ${loss}