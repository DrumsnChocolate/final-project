from finetune.datasets.ade import ADE20KMaskDataset
from finetune.datasets.cbis import CBISMultiMaskDataset, CBISBinaryMaskDataset
from finetune.datasets.zgt import ZGTBinaryMaskDataset
from finetune.datasets.segmentation_dataset import SegmentationMaskDataset

dataset_class_mapping = {
    'ade20k': ADE20KMaskDataset,
    'cbis-binary': CBISBinaryMaskDataset,
    'cbis-multi': CBISMultiMaskDataset,
    'zgt-binary': ZGTBinaryMaskDataset,
}

class SegmentationMaskLoader:
    def __init__(self, cfg, dataset: SegmentationMaskDataset):
        self.cfg = cfg
        self.dataset = dataset

    def __getitem__(self, item):
        assert isinstance(item, int), f'only able to provide items by index, not with argument {item}'
        if item == len(self):
            raise IndexError(f'index {item} out of range')
        start_index = item * self.cfg.data[self.dataset._split].batch_size
        end_index = min(start_index + self.cfg.data[self.dataset._split].batch_size, len(self.dataset))
        return self.dataset[start_index:end_index]

    def __len__(self):
        # double negative to round up instead of down https://stackoverflow.com/a/35125872
        return -(- len(self.dataset) // self.cfg.data[self.dataset._split].batch_size)


def build_dataloaders(cfg):
    loaders = {}
    splits = ['train', 'val', 'test']
    for split in splits:
        if cfg.data.get(split) is None:
            continue
        dataset_class = dataset_class_mapping[cfg.data.name]
        dataset = dataset_class(cfg, split)
        loaders[split] = SegmentationMaskLoader(cfg, dataset)
        # if cfg.data.name == 'ade20k':
        #     dataset = ADE20KMaskDataset(cfg, split)
        # if cfg.data.name == 'cbis-binary':
        #     dataset = CBISBinaryMaskDataset(cfg, split)
        # if cfg.data.name == 'cbis-multi':
        #     dataset = CBISMultiMaskDataset(cfg, split)
        # if cfg.data.name == 'zgt-binary':
        #     dataset = ZGTBinaryMaskDataset(cfg, split)
        # loaders[split] = SegmentationMaskLoader(cfg, dataset)
    return loaders



# def build_dataloaders(cfg, mask=True):
#     if mask:
#         if cfg.data.name == 'ade20k':
#             train_dataset = ADE20KMaskDataset(cfg, 'train')
#             val_dataset = ADE20KMaskDataset(cfg, 'val')
#             test_dataset = ADE20KMaskDataset(cfg, 'test')
#         elif cfg.data.name == 'cbis-binary':
#             train_dataset = CBISBinaryMaskDataset(cfg, 'train')
#             val_dataset = CBISBinaryMaskDataset(cfg, 'val')
#             test_dataset = CBISBinaryMaskDataset(cfg, 'test')
#         elif cfg.data.name == 'cbis-multi':
#             train_dataset = CBISMultiMaskDataset(cfg, 'train')
#             val_dataset = CBISMultiMaskDataset(cfg, 'val')
#             test_dataset = CBISMultiMaskDataset(cfg, 'test')
#         else:
#
#             raise NotImplementedError()
#     else:
#             raise NotImplementedError()
#
#     return {
#         'train': SegmentationMaskLoader(cfg, train_dataset),
#         'val': SegmentationMaskLoader(cfg, val_dataset),
#         'test': SegmentationMaskLoader(cfg, test_dataset),
#     }
