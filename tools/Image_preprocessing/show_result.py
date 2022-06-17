import shapely.geometry
import shapely.ops
from pycocotools.coco import COCO
from .save_read_geotiff import get_image_info
import json
import torch
import os
from osgeo import gdal
import numpy as np
import tqdm
from mmcv.ops.nms import batched_nms, nms_match
import copy
from tools.utils import stdout_off, stdout_on
import cv2
from pycocotools.coco import COCO
from skimage import io
from matplotlib import pyplot as plt
import math

def union_segm(
        js_data,
        nms_cfg,
        ori_img_path,
        ref_json,
        merge_cats=False,
        score_thr=[0.85],
        save_path=None,

):
    def __as_polygon(segm):
        segms = [shapely.geometry.asPolygon(np.array(s, dtype=np.float).reshape(-1, 2).tolist()) for s in segm if len(s)>=6]
        areas = [s.area for s in segms]
        idx = np.argmax(areas)
        polygon = segms[idx]
        return polygon

    stdout_on()
    res_poly = dict()
    polys = []
    shp_path = os.path.join(save_path,"seg")
    os.makedirs(shp_path, mode = 0o777, exist_ok = True)
    for j in js_data:
        poly = __as_polygon(j['segmentation'])
        polys.append(poly)
    score = np.array([s['score'] for s in js_data])
    for st in score_thr:
        idxs = score >= st
        polygons = shapely.ops.unary_union([p for i, p in zip(idxs, polys) if i])
        res_poly[st] = shapely.geometry.mapping(polygons)
        # if res_poly[st]['type'] is 'Polygon':
        #     res_poly = (res_poly)

    for st, geojs in res_poly.items():
        coco = COCO(ref_json)
        str = os.path.splitext(os.path.split(ori_img_path)[1])[0]
        image = io.imread(ori_img_path)
        seg = [
            {
                "segmentation": [
                ]
            }
        ]
        for idx in range(len(geojs['coordinates'])):

            points  = list(geojs['coordinates'][idx][0])
            polygon = []
            for point in points:
                polygon.append(point[0])
                polygon.append(point[1])

            seg[0]["segmentation"].append(polygon)
        fig, ax = plt.subplots()
        ax.imshow(image)
        coco.showAnns(seg)
        plt.axis("off")
        height, width, channels = image.shape

        fig.set_size_inches(width / 100.0, height / 100.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)
        save_file = os.path.join(shp_path, str + f"union_segm_{st}.tif")
        plt.savefig(save_file)
        plt.close(fig)
    return save_file



def detect_result_to_json(
        res_js_file,
        dataset_js_file,
        dataset_img_path,
):
    cocoGt = COCO(dataset_js_file)

    try:
        cocoDt = cocoGt.loadRes(res_js_file)
    except IndexError:
        print('The testing results of the whole dataset is empty.')
        cocoDt = COCO()
    im_geotrans = dict()
    for i, img in cocoDt.imgs.items():
        img_file = dataset_img_path
        _, _, _, im_geotran, _ = get_image_info(img_file)
        im_geotrans[i] = list(im_geotran)

    res_json = []
    for i, ann in cocoDt.anns.items():
        img_id = ann["image_id"]
        bbox = ann["bbox"]

        if ann.get("segmentation", None) is None:
            json_temp = dict(
                image_id=img_id,
                bbox=bbox,
                score=ann["score"],
                category_id=ann["category_id"]
            )
        else:
            mask = cocoDt.annToMask(ann)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            polygon = [cont.flatten() for cont in contours]

            geo_segm = []
            for p in polygon:
                xs = p[::2]
                ys = p[1::2]
                segm = np.array([[x, y] for x, y in zip(xs, ys)]).flatten().tolist()
                geo_segm.append(segm)
            json_temp = dict(
                image_id=img_id,
                bbox=bbox,
                score=ann["score"],
                category_id=ann["category_id"],
                segmentation=geo_segm
            )
        res_json.append(json_temp)

    return res_json


def show_result(
    res_js_file,
    dataset_js_file,
    dataset_img_path,
    ori_img_path,
    ref_json,
    nms_merge_cats,
    nms_iou_thr = 0.5,
    score_thr = [0.85],

):
    result_json = detect_result_to_json(
        res_js_file,
        dataset_js_file,
        dataset_img_path
    )

    # NMS
    print("3.NMS.", end=' ')
    stdout_off()
    nms_cfg = dict(
        type='nms',
        iou_threshold=nms_iou_thr
    )

    seg_path = union_segm(
        js_data=result_json,
        nms_cfg=nms_cfg,
        merge_cats=nms_merge_cats,
        score_thr=score_thr,
        save_path=os.path.split(res_js_file)[0],
        ori_img_path=ori_img_path,
        ref_json = ref_json
    )

    return seg_path

