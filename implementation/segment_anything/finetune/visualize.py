import argparse
import os

import torch
from torchvision.io import read_image, ImageReadMode, write_png

from finetune.checkpoint import contains_loadable_model, load
from finetune.configs.config_options import DictAction
from finetune.configs.config_validation import validate_cfg_test
from finetune.datasets.loaders import build_dataloaders
from finetune.datasets.segmentation_dataset import ensure_image_rgb
from finetune.loss import build_loss_function
from finetune.metrics import build_metric_functions
from matplotlib import pyplot as plt
from finetune.models.build_model import build_model
from finetune.models.sam_wrapper import SamWrapper
from finetune.test import test_epoch, get_test_logger
from finetune.train import get_cfg, store_cfg, get_point_prompts


def parse_args():
    parser = argparse.ArgumentParser(description='Apply a segment anything model to a subset of the validation data, and save the visualized input prompts + outputs')
    parser.add_argument('config', help='config file containing the instructions that the model was trained/zero-shot with.')
    parser.add_argument('--output_dir', help='relative path to directory to save the visualizations', required=True)
    parser.add_argument('--image_names', nargs='+', help='names of the images to visualize')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
    )
    return parser.parse_args()


def store_visualization(cfg, model, point_prompts, predicted_masks, predicted_ious, image_name, dataloader):
    visualization_dir = cfg.visualization_dir
    image_path = dataloader.dataset.get_image_path(image_name)
    original_image = ensure_image_rgb(read_image(image_path))

    preprocessed_dimensions = predicted_masks.shape[-2:]  # predicted masks shape is MxHxW
    original_dimensions = original_image.shape[-2:]  # original image shape is CxHxW
    # resize each predicted mask to the original image size
    predicted_masks = torch.nn.functional.interpolate(predicted_masks.unsqueeze(1), size=original_dimensions, mode='nearest').squeeze(1)
    # also recalculate the point prompt
    point_prompts = (point_prompts / torch.tensor(preprocessed_dimensions, device=cfg.device).float() * torch.tensor(original_dimensions, device=cfg.device).float()).round().int()
    best_index = torch.argmax(predicted_ious)
    best_mask = (predicted_masks[best_index] > model.mask_threshold) * 1
    os.makedirs(visualization_dir, exist_ok=True)
    write_png(torch.tensor(best_mask*255, dtype=torch.uint8, device='cpu').unsqueeze(0), os.path.join(visualization_dir, f'{image_name}.png'), compression_level=0)
    with open(os.path.join(visualization_dir, f'{image_name}.txt'), 'w') as f:
        f.write(f'{point_prompts.to("cpu").tolist()}')




def visualize_samples(cfg, model: SamWrapper, dataloader, image_names=None):
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            samples, targets, classes = batch
            point_prompts, point_prompts_labels = get_point_prompts(targets)
            predicted_masks, predicted_ious, _ = model(samples, point_prompts, point_prompts_labels)
            for j in range(len(samples)):
                image_name = image_names[i*cfg.data.val.batch_size + j]
                store_visualization(cfg, model, point_prompts[j], predicted_masks[j], predicted_ious[j], image_name, dataloader)


def visualize(cfg, image_names=None):
    cfg.data.train = None
    if cfg.data.get('val') is None:
        cfg.data.val = cfg.data.test
    cfg.data.test = None
    dataloaders = build_dataloaders(cfg, image_names=image_names)
    model = build_model(cfg, None)
    if contains_loadable_model(cfg):
        model, _ = load(cfg, model)
    visualize_samples(cfg, model, dataloaders['val'], image_names=image_names)


def main():
    args = parse_args()
    cfg = get_cfg(args, validate_cfg=validate_cfg_test)
    cfg.sub_dir = 'test'
    assert args.output_dir is not None, 'output_dir must be set'
    cfg.visualization_dir = args.output_dir
    visualize(cfg, image_names=args.image_names)


if __name__ == '__main__':
    main()
