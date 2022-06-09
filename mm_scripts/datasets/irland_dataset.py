
from .coco import COCOFormatDataset
from mmdet.datasets.builder import DATASETS


@DATASETS.register_module()
class IrLandDataset(COCOFormatDataset):

    CLASSES = ('crops_completed_circle', 'crops_uncompleted_polygon')


@DATASETS.register_module()
class IrLandOneCatDataset(COCOFormatDataset):

    CLASSES = ('crops_completed_circle',)
