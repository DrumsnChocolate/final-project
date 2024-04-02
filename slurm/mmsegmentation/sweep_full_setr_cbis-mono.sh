lr=0.0001
size=224
batch_size=16
pretrained=true
loss="cross_entropy"
sbatch -J "full-setr-cbis-mono" slurm/mmsegmentation/setr_cbis_mono.sbatch "${lr}" "${size}" "${batch_size}" "${pretrained}" "${loss}"
