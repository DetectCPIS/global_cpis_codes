from mmcv import Config
from mmcv.runner import init_dist

from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)


def build_mmdet_dataset(
        cfg_file,
        img_dir,
        json_path,
        img_scale
):
    launcher = 'none'

    cfg = Config.fromfile(cfg_file)
    cfg.data.test["ann_file"] = json_path
    cfg.data.test["img_prefix"] = img_dir
    test_pipeline = cfg.data.test["pipeline"]
    for p in test_pipeline:
        if p['type'] == "MultiScaleFlipAug":
            p["img_scale"] = img_scale
            trans = p["transforms"]
            for t in trans:
                if t['type'] == 'Resize':
                    t['img_scale'] = img_scale

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(launcher, **cfg.dist_params)

    # build the dataloader
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    return data_loader


