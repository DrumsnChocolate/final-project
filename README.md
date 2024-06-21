# Final Project
Implementation of experiments related to my Master's Thesis

The following repositories have been adapted into this project (see the folder `implementation` for the adapted code):
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) has been used as the basis for the segmentation transformer runs. We've adapted their code to function with binary segmentation maps too.
- [Segment Anything](https://github.com/facebookresearch/segment-anything) contained the Segment Anything Model that we used. However, only demonstrating code was included, so we've had to supply our own training and testing logic. Our additions were mostly made in the `implementation/segment_anything/finetune` folder.
- [Visual Prompt Tuning](https://github.com/KMnP/vpt) has been used as the basis for Visual Prompt Tuning for the Segmentation Transformer.
- [Connected-Unets-and-more](https://github.com/AsmaBaccouche/Connected-Unets-and-more) has not been used in our experiments, but we've taken inspiration from the implementation for some of their metrics.
