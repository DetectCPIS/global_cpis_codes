
from mmdet.datasets.coco import CocoDataset
import time
import numpy as np
import os

import itertools
import logging
from collections import OrderedDict

from mmcv.utils import print_log
from pycocotools.cocoeval import COCOeval
from terminaltables import AsciiTable

from tools.evaluation import eval_file


class COCOFormatDataset(CocoDataset):

    CLASSES = ()

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        """


        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'mAP', 'recall'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple], optional): Scale ranges for evaluating
                mAP. If not specified, all bounding boxes would be included in
                evaluation. Default: None.

        Returns:
            dict[str, float]: AP/recall metrics.
        """

        # if jsonfile_prefix is not None:
        #     jsonfile_prefix = jsonfile_prefix+f"_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"

        # Check input.
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        # Save result.
        if jsonfile_prefix is not None:
            os.makedirs(os.path.split(jsonfile_prefix)[0], mode=0o777, exist_ok=True)
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        # Evaluation.
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')

            eval_file(self.ann_file,
                     result_files[metric],
                     metric=metric)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        eval_results = super().evaluate(
            results,
            metric=metric,
            logger=logger,
            jsonfile_prefix=jsonfile_prefix,
            classwise=classwise,
            proposal_nums=proposal_nums,
            iou_thrs=iou_thrs,
            metric_items=metric_items)

        return eval_results
