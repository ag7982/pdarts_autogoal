from torch.utils.data import DataLoader
from autogoal.grammar import *
from autogoal.utils import nice_repr

from ..preprocessing import Preprocessor
from ..pdarts import PDarts

@nice_repr
class Pipeline:

    def __init__(
        self,
        preprocessing: Preprocessor,
        pdarts: PDarts,
        batch_size: DiscreteValue(60, 96),

    ):

        self.preprocessing = preprocessing
        self.pdarts = pdarts
        self.batch_size = batch_size

    
    def fit(self, train_dataset, valid_dataset):
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=1
        )

        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=1
        )

        train_transform, valid_transform = self.preprocessing.fit(train_dataset)

        train_dataset.transform = train_transform
        valid_dataset.transform = valid_transform

        return self.pdarts.fit(train_loader, valid_loader)