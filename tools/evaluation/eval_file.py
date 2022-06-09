
import os
import itertools
import numpy as np
from pycocotools.coco import COCO
from .cocoeval import EvalByRecall, EvalByScore
from terminaltables import AsciiTable


def eval_file(gt_file,
              res_file,
              metric="bbox",
              catid=-1,
              max_det=10000,
              iou_thrs=0.5,
              area_rng=[0, 1e10],
              score_thr=[0.01*i for i in range(50, 100, 5)],
              file_prefix=None,
              ):

    if file_prefix is None:
        file_prefix = os.path.splitext(res_file)[0]
    if isinstance(area_rng, list) and not isinstance(area_rng[0], list):
        area_rng = [area_rng]
    if not isinstance(max_det, list):
        max_det = [max_det//100, max_det//10, max_det]
    if not isinstance(iou_thrs, list):
        iou_thrs = [iou_thrs]
    if not isinstance(score_thr, list):
        score_thr = [score_thr]

    cocoGt = COCO(gt_file)
    try:
        cocoDt = cocoGt.loadRes(res_file)
    except IndexError:
        print('The testing results of the whole dataset is empty.')

    area_reg_lbl = [f"{a}" for a in area_rng]
    catIds = cocoGt.get_cat_ids() if catid == -1 else catid
    imgIds = cocoGt.get_img_ids()
    iou_type = 'bbox' if metric == 'proposal' else metric
    scoEval = EvalByScore( score_thrs=score_thr,
                            areaReg=area_rng,
                            areaRegLbl=area_reg_lbl,
                            cocoGt=cocoGt,
                            cocoDt=cocoDt,
                            iouType=iou_type)
    scoEval.params.catIds = catIds
    scoEval.params.imgIds = imgIds
    scoEval.params.maxDets = max_det
    scoEval.params.iouThrs = iou_thrs
    scoEval.evaluate()
    scoEval.accumulate()

    recEval = EvalByRecall( cocoGt=cocoGt,
                            cocoDt=cocoDt,
                            iouType=iou_type)
    recEval.params.catIds = catIds
    recEval.params.imgIds = imgIds
    recEval.params.maxDets = max_det
    recEval.evaluate()
    recEval.accumulate()
    recEval.summarize()

    if True:  # Compute per-category AP
        # Compute per-category AP
        # from https://github.com/facebookresearch/detectron2/
        precisions = recEval.eval['precision']
        # precision: (iou, recall, cls, area range, max dets)
        assert len(catIds) == precisions.shape[2]

        results_per_category = []
        for idx, catId in enumerate(catIds):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            nm = cocoGt.loadCats(catId)[0]
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            if precision.size:
                ap = np.mean(precision)
            else:
                ap = float('nan')
            results_per_category.append(
                (f'{nm["name"]}', f'{float(ap):0.3f}'))

        num_columns = min(6, len(results_per_category) * 2)
        results_flatten = list(
            itertools.chain(*results_per_category))
        headers = ['category', 'AP'] * (num_columns // 2)
        results_2d = itertools.zip_longest(*[
            results_flatten[i::num_columns]
            for i in range(num_columns)
        ])
        table_data = [headers]
        table_data += [result for result in results_2d]
        table = AsciiTable(table_data)
        print('\n' + table.table )

    pr_file_name = file_prefix + \
                   f"_iouthr{iou_thrs}" \
                   f"_areaRng{area_rng}" \
                f"_maxDet{max_det}"
    with open(pr_file_name +".csv", "w") as f:
        # score Eval.
        print(",", end=" ", file=f)
        for i in catIds:
            print(f"{cocoGt.cats[i]['name']}, ," ,end=' ', file=f)
        print("", file=f)

        print("score_thr,", end=" ", file=f)
        for i in catIds:
            print(f"precision, recall,", end=' ', file=f)
        print("", file=f)

        for i, s in enumerate(score_thr):
            print(f"{s},", end=" ", file=f)
            for j in catIds:
                print(f"{scoEval.eval['precision'][0, i, j, 0, 0]}, "
                      f"{scoEval.eval['recall'][0, i, j, 0, 0]},", end=' ', file=f)
            print("", file=f)
        print("", file=f)

        # recall eval
        for i in catIds:
            print(f"{cocoGt.cats[i]['name']}", file=f)
            print(f"recall", file=f, end=',')
            for r in recEval.params.recThrs:
                print(f"{r}", file=f, end=',')
            print("", file=f)
            for iou_idx, iou_thr in enumerate(recEval.params.iouThrs):
                print("pre(iou>{:.2})".format(iou_thr), file=f, end=',')
                for p in recEval.eval["precision"][iou_idx, :, i, 0, 0]:
                    print(f"{p}", file=f, end=',')
                print("", file=f)
        print("", file=f)

        # mAP
        print(", IoU, area, maxDets, val", file=f)
        print(f"AP, 0.50:0.95, all, 100, {recEval.stats[0]}", file=f)
        print(f"AP, 0.50, all, 100, {recEval.stats[1]}", file=f)
        print(f"AP, 0.75, all, 100, {recEval.stats[2]}", file=f)
        print(f"AP, 0.50:0.95, small, 100, {recEval.stats[3]}", file=f)
        print(f"AP, 0.50:0.95, medium, 100, {recEval.stats[4]}", file=f)
        print(f"AP, 0.50:0.95, large, 100, {recEval.stats[5]}", file=f)
        print(f"AR, 0.50:0.95, all, 1, {recEval.stats[6]}", file=f)
        print(f"AR, 0.50:0.95, all, 10, {recEval.stats[7]}", file=f)
        print(f"AR, 0.50:0.95, all, 100, {recEval.stats[8]}", file=f)
        print(f"AR, 0.50:0.95, small, 100, {recEval.stats[9]}", file=f)
        print(f"AR, 0.50:0.95, medium, 100, {recEval.stats[10]}", file=f)
        print(f"AR, 0.50:0.95, large, 100, {recEval.stats[11]}", file=f)



