lr=0.0001
size=224
batch_size=16
pretrained=false
loss="cross_entropy"
tuning_method=full
sbatch -J "full-setr-cbis-mono" slurm/mmsegmentation/setr_cbis_mono.sbatch "${lr}" "${size}" "${batch_size}" "${pretrained}" "${loss}" "${tuning_method}"
