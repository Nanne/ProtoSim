import torch
from torchvision import datasets, transforms

class ReturnIndexWrapper(object):

    def __init__(self, dataset, index_labels=False):
        self._inner = dataset
        self.index_labels = index_labels

    def __getitem__(self, idx):
        img, lab = self._inner.__getitem__(idx)
        if self.index_labels:
            return img, idx, lab
        else:
            return img, idx

    def __len__(self):
        return self._inner.__len__()

    def __getattr__(self, attr):
        if attr in self.__class__.__dict__:
            return getattr(self, attr)
        else:
            return getattr(self._inner, attr)

    def __setattr__(self, attr, value):
        if attr in self.__class__.__dict__ or attr in ['_inner']:
            super(ReturnIndexWrapper, self).__setattr__(attr, value)
        else:
            return self._inner.__setattr__(attr, value)

def build_dataset(data_path, transform, indexed=False, index_labels=False):
    dsets = []

    for ds in data_path.split(","):
        dsets.append(datasets.ImageFolder(ds, transform=transform))

    dataset = torch.utils.data.ConcatDataset(dsets)
    nb_classes = sum([len(d.classes) for d in dsets])
    if indexed:
        dataset = ReturnIndexWrapper(dataset, index_labels=index_labels)

    return dataset, nb_classes
