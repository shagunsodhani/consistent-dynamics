import torch

from codes.data.loader.loaders import RolloutSequenceDataset as dataset_cls


def get_dataloaders(config, modes = ("train", "val", "test")):
    return dict(list(map(lambda mode: (mode, _get_dataloader(config, mode)), modes)))


def _get_dataloader(config, mode="train"):
    batch_size = config.dataset.batch_size
    if mode!="train":
        batch_size = int((batch_size+1)/2)
    shuffle = True
    num_workers = config.dataset.num_workers
    dataset = dataset_cls(config, mode)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
