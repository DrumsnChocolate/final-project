lr=0.0001
size=512
batch_size=12
sbatch -J "full-setr-cbis-mono" slurm/mmsegmentation/setr_cbis_mono.sbatch "${lr}" "${size}" "${batch_size}"
