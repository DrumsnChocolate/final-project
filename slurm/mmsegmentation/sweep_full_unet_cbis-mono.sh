lr=0.0001
size=224
loss="cross_entropy"
clahe="true"
augmentation="false"
mass_only="true"
sbatch -J "full-unet-cbis-mono-lr${lr}" slurm/mmsegmentation/unet_cbis.sbatch "${lr}" "${size}" "${loss}" "${clahe}" "${augmentation}" "${mass_only}"
