import os
from tools.detect_scripts import detect_sentinel_batch

if __name__ == '__main__':
    model_cfg = dict(
        cfg_file="model/dt_cascade_pointrend_x101_32x4d.py",
        checkpoint="model/dt_cascade_pointrend_x101_32x4d.pth",
    )
    preprocess_cfg = dict(
        ref_dataset_json="model/ann.json",
    )
    result_merge_cfg = dict(
        nms_thr=0.1,
        nms_merge_cats=True,
        score_thr=[0.85],
    )

    work_dir = "test"
    ori_img_dir = "imgs"
    detect_sentinel_batch(
        ori_img_dir=ori_img_dir,
        img_list_file=os.listdir(ori_img_dir),
        workdir="temp",
        seg_res_path="result",
        model_cfg=model_cfg,
        **preprocess_cfg,
        **result_merge_cfg,
    )

    pass

