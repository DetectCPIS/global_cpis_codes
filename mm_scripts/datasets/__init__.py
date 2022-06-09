"""
Register private datasets.

"""
from .irland_dataset import IrLandDataset, IrLandOneCatDataset
from .pipelines import *


__all__ = ['IrLandDataset', 'IrLandOneCatDataset']
print("./mm_scripts/datasets registered.")
