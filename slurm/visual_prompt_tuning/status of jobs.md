jobs that don't quite work yet:

- 

Possible solutions to jobs that failed:

- [x] resisc45 needs manual downloading, put it in visual_prompt_tuning/data_path/downloads/manual
- [x] dtd needs more gpu mem, unclear how much exactly. try scaling up to 2 gpus
- [x] cifar needs more gpu mem. try 2 gpus
- [x] kitti 3.2.0 is too old, need newer version
- [x] oxford_flowers needs more gpu mem. try 2 gpus?
- [x] oxford_iiit_pet needs more gpu mem. try 2 gpus?
- [x] same for svhn
- [x] same for camelyon
- [x] same for smallnorb label elevation
- [x] same for dmlab
- [x] same for clevr count
- [x] same for clevr closest
- [x] same for dsprites label_orientation
- [x] same for dsprites label x position
- [x] diabetic retinopathy still needs to be placed in the manual folder
- [x] something is going on with sun397, not sure what. It has to do with the data
  okay, this had to do with: https://github.com/tensorflow/datasets/issues/2889 which was fixed by doing the same as in https://github.com/tensorflow/datasets/pull/4955





- [x] oxford_flowers needs more gpu mem. try 3 gpus
- [x] dtd same
- [x] oxford_iiit_pet same
- [x] cifar same
- [x] camelyon same
- [x] svhn same
- [x] dmlab same
- [x] resisc45 same
- [x] smallnorb label elevation same
- [x] kitti same
- [x] dsprites same

dsprites jobs were killed, for some reason:

- [ ] dsprites x_position 235568 was killed in four out of 5 runs
- [ ] 