example usage:

(make sure you are running from the root of the segment_anything folder)
```bash
conda activate segment_anything
python finetune/train.py finetune/configs/test.yaml --cfg-options model.finetuning.name=bier
```
This will use the test config file, and override it with the options passed via --cfg-options