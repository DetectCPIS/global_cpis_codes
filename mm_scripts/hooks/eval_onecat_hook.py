from mmcv.runner.hooks import HOOKS, Hook
from tools.evaluation import eval_file
from tools.result_preprocessing import merge_category

import os


@HOOKS.register_module()
class EvalOnecatHook(Hook):

    def __init__(self, ann_file):
        print("EvalOnecatHook builded.")
        self.ann_file = ann_file

    def after_epoch(self, runner):
        print("Eval one category result.")

        workdir = runner.work_dir
        epoch = runner.epoch
        metrics = ["bbox", "segm"]
        for metric in metrics:
            file = os.path.join(workdir, f"result_epoch_{epoch}.{metric}.json")
            if not os.path.exists(file):
                continue
            onecat_file = merge_category(result_json=file)

            eval_file(
                gt_file=self.ann_file,
                res_file=onecat_file,
                metric=metric,
                catid=-1,
                max_det=10000,
                iou_thrs=0.5,
                area_rng=[0, 1e10],
                score_thr=[0.01 * i for i in range(50, 100, 5)],
            )

        pass