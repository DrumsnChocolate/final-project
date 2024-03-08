lr=0.0001
size=256
sbatch -J "full-setr-cbis-mono" slurm/mmsegmentation/setr_cbis_mono.sbatch "${lr}" "${size}"
