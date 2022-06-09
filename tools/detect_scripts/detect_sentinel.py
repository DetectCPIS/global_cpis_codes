"""

"""

import os
import shutil
from tools.sentinel_scripts import detect_dataset
from tools.Image_preprocessing import show_result
from osgeo import gdal
from tools.Image_preprocessing.generate_dataset_json import generate_test_json



def detect_sentinel(
        img_path,
        ref_json,
        work_dir,
        nms_thr,
        nms_merge_cats,
        model_cfg,
        model=None,
        score_thr=None,
):
    img_name = os.path.splitext(os.path.split(img_path)[1])[0]
    print(f"===================")
    print(f"Detect image {img_name}")
    print()

    os.makedirs(os.path.abspath(work_dir), mode=0o777, exist_ok=True)
    # img_name = os.path.splitext(os.path.split(img_path)[1])[0]
    img_work_dir = os.path.join(work_dir, img_name)
    root_ds = gdal.Open(img_path)
    im_width = root_ds.RasterXSize
    im_height =root_ds.RasterYSize

    json_path = os.path.join(img_work_dir, "annotations")
    os.makedirs(json_path, mode=0o777, exist_ok=True)
    generate_test_json(
        img_file=img_path,
        ref_json=ref_json,
        out_path=json_path
    )
    print("done.")
    print()

    # detect
    img_dir = os.path.join(img_work_dir, "images")
    os.makedirs(img_dir, mode=0o777, exist_ok=True)
    image_path = os.path.join(img_dir,os.path.split(img_path)[1])
    shutil.copyfile(img_path, image_path)
    json_path = os.path.join(img_work_dir, "annotations/test.json")
    sub_img_res_json_path = os.path.join(img_work_dir, "sub_img_result.json")
    dataset_cfg = dict(
        cfg_file=model_cfg["cfg_file"],
        img_dir=img_dir,
        json_path=json_path,
        img_scale=(im_width, im_height)
    )
    detect_dataset(
        model=model if model is not None else model_cfg,
        dataset=dataset_cfg,
        out_file=sub_img_res_json_path
    )
    print()

    # union result
    seg_res_path = show_result(
        res_js_file=sub_img_res_json_path,
        dataset_js_file=json_path,
        dataset_img_path=image_path,
        ori_img_path=img_path,
        ref_json=ref_json,
        nms_iou_thr=nms_thr,
        nms_merge_cats=nms_merge_cats,

    )

    print()


    return seg_res_path
