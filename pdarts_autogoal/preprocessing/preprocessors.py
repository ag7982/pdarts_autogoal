from torchvision.transforms import (
    GaussianBlur as _GaussianBlur,
    RandomAffine as _RandomAffine,
    RandomHorizontalFlip as _RandomHorizontalFlip,
    RandomVerticalFlip as _RandomVerticalFlip,
    RandomErasing as _RandomErasing,
    ColorJitter as _ColorJitter
)
from autogoal.grammar import *


class RandomAffine(_RandomAffine):

    def __init__(
        self,
        degrees: ContinuousValue(0, 360),
        translate_w: ContinuousValue(0, 1),
        translate_h: ContinuousValue(0, 1),
        scale_min: ContinuousValue(0, 1),
        scale_range: ContinuousValue(1, 10),
        shear: ContinuousValue(0, 360),
        is_enabled: BooleanValue()
    ):
        super().__init__(
            degrees,
            translate=(translate_w, translate_h),
            scale=(scale_min, scale_min + scale_range),
            shear=shear
        )

        self.is_enabled = is_enabled

class GaussianBlur(_GaussianBlur):

    def __init__(
        self, 
        kernel_size: DiscreteValue(0, 20),
        sigma: ContinuousValue(0.1, 2.0),
        is_enabled: BooleanValue()
    ):
        super().__init__(kernel_size=kernel_size*2+1, sigma=sigma)
        self.is_enabled = is_enabled

class RandomHorizontalFlip(_RandomHorizontalFlip):

    def __init__(
        self,
        p: ContinuousValue(0, 1),
        is_enabled: BooleanValue()
    ):
        super().__init__(p=p)
        self.is_enabled = is_enabled

class RandomVerticalFlip(_RandomVerticalFlip):

    def __init__(
        self,
        p: ContinuousValue(0, 1),
        is_enabled: BooleanValue()
    ):
        super().__init__(p=p)
        self.is_enabled = is_enabled

class ColorJitter(_ColorJitter):

    def __init__(
        self,
        brightness: ContinuousValue(0, 20),
        contrast: ContinuousValue(0, 20),
        saturation: ContinuousValue(0, 20),
        hue: ContinuousValue(0, 0.5),
        is_enabled: BooleanValue()
    ):
        super().__init__(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
        self.is_enabled = is_enabled

class RandomErasing(_RandomErasing):

    def __init__(
        self,
        p: ContinuousValue(0, 1.0),
        is_enabled: BooleanValue()
    ):
        super().__init__(p=p)
        self.is_enabled = is_enabled