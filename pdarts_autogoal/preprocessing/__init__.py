import torch
from torchvision.transforms import RandomCrop, Normalize, ToTensor, Compose, Grayscale
from torch.utils.data import DataLoader

from autogoal.grammar import *
from autogoal.utils import nice_repr

from .preprocessors import *

@nice_repr
class Preprocessor:

    def __init__(
        self,
        blur: GaussianBlur,
        affine: RandomAffine,
        h_flip: RandomHorizontalFlip,
        v_flip: RandomVerticalFlip,
        cutout: RandomErasing,
        jitter: ColorJitter,
        norm: BooleanValue(),
        crop: BooleanValue()
    ):
        self.blur = blur
        self.affine = affine
        self.h_flip = h_flip
        self.v_flip = v_flip
        self.cutout = cutout
        self.jitter = jitter
        self.norm = norm
        self.crop = crop

    def _get_stats(self, dataset):
        dataset.transform = Compose([ Grayscale(3),ToTensor()])
        loader = DataLoader(
            dataset, 
            batch_size=100, 
            num_workers=1
        )
        h, w = 0, 0
        for batch_idx, (inputs, targets) in enumerate(loader):
            if batch_idx == 0:
                h, w = inputs.size(2), inputs.size(3)
                chsum = inputs.sum(dim=(0, 2, 3), keepdim=True)
            else:
                chsum += inputs.sum(dim=(0, 2, 3), keepdim=True)
        mean = chsum/len(dataset)/h/w

        chsum = None
        for batch_idx, (inputs, targets) in enumerate(loader):
            if batch_idx == 0:
                chsum = (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
            else:
                chsum += (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
        std = torch.sqrt(chsum/(len(dataset) * h * w - 1))

        return mean.squeeze(0), std.squeeze(0), h, w

    
    def fit(self, train_dataset):
        mean, std, h, w = self._get_stats(train_dataset)

        train_transform = [
#             Grayscale(3),
            self.blur if self.blur.is_enabled else None,
            self.affine if self.affine.is_enabled else None, 
            self.h_flip if self.h_flip.is_enabled else None,
            self.v_flip if self.v_flip.is_enabled else None, 
            self.jitter if self.jitter.is_enabled else None,
            ToTensor(),
            self.cutout if self.cutout.is_enabled else None,
            Normalize(mean, std) if self.norm else None
        ]

        valid_transform = [
            Grayscale(3),
            ToTensor(),
            Normalize(mean, std) if self.norm else None
        ]
        
        print([i for i in train_transform if i is not None])

        return (
            Compose([i for i in train_transform if i is not None]),
            Compose([i for i in valid_transform if i is not None])
        )
    

