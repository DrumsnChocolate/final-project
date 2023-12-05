from finetune.datasets.ade import ADE20KDataset
from finetune.datasets.cbis import CBISMultiDataset, CBISBinaryDataset
from finetune.datasets.segmentation_dataset import SegmentationDataset


class SegmentationLoader:
    def __init__(self, cfg, dataset: SegmentationDataset):
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
        train_dataset = ADE20KDataset(cfg, 'train')
        val_dataset = ADE20KDataset(cfg, 'val')
        test_dataset = ADE20KDataset(cfg, 'test')
    elif cfg.data.name == 'cbis-binary':
        train_dataset = CBISBinaryDataset(cfg, 'train')
        val_dataset = CBISBinaryDataset(cfg, 'val')
        test_dataset = CBISBinaryDataset(cfg, 'test')
    elif cfg.data.name == 'cbis-multi':
        train_dataset = CBISMultiDataset(cfg, 'train')
        val_dataset = CBISMultiDataset(cfg, 'val')
        test_dataset = CBISMultiDataset(cfg, 'test')
    else:
        raise NotImplementedError()

    return {
        'train': SegmentationLoader(cfg, train_dataset),
        'val': SegmentationLoader(cfg, val_dataset),
        'test': SegmentationLoader(cfg, test_dataset),
    }
