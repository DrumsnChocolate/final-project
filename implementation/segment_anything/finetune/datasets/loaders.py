from finetune.datasets.ade import ADE20KMaskDataset
from finetune.datasets.cbis import CBISMultiMaskDataset, CBISBinaryMaskDataset
from finetune.datasets.segmentation_dataset import SegmentationDataset, SegmentationMaskDataset


class SegmentationLoader:
    def __init__(self, cfg, dataset: SegmentationDataset | SegmentationMaskDataset):
        self.cfg = cfg
        self.dataset = dataset

    def __getitem__(self, item):
        assert isinstance(item, int), f'only able to provide items by index, not with argument {item}'
        start_index = item * self.cfg.data[self.dataset._split].batch_size
        end_index = min(start_index + self.cfg.data[self.dataset._split].batch_size, len(self.dataset))
        return self.dataset[start_index:end_index]

    def __len__(self):
        # double negative to round up instead of down https://stackoverflow.com/a/35125872
        return -(- len(self.dataset) // self.cfg.data[self.dataset._split].batch_size)





def build_dataloaders(cfg):
    if cfg.data.name == 'ade20k':
        train_dataset = ADE20KMaskDataset(cfg, 'train')
        val_dataset = ADE20KMaskDataset(cfg, 'val')
        test_dataset = ADE20KMaskDataset(cfg, 'test')
    elif cfg.data.name == 'cbis-binary':
        train_dataset = CBISBinaryMaskDataset(cfg, 'train')
        val_dataset = CBISBinaryMaskDataset(cfg, 'val')
        test_dataset = CBISBinaryMaskDataset(cfg, 'test')
    elif cfg.data.name == 'cbis-multi':
        train_dataset = CBISMultiMaskDataset(cfg, 'train')
        val_dataset = CBISMultiMaskDataset(cfg, 'val')
        test_dataset = CBISMultiMaskDataset(cfg, 'test')
    else:
        raise NotImplementedError()

    return {
        'train': SegmentationLoader(cfg, train_dataset),
        'val': SegmentationLoader(cfg, val_dataset),
        'test': SegmentationLoader(cfg, test_dataset),
    }
