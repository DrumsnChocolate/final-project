lr=0.0001
size=512
sbatch -J "full-unet-cbis-mono-lr${lr}" slurm/mmsegmentation/unet_cbis.sbatch ${lr} ${size}