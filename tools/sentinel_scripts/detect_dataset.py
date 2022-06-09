import json
import torch
import os
from .build_dataset import build_mmdet_dataset
from .build_model import build_mmdet_model
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmdet.apis import multi_gpu_test, single_gpu_test
from tools.utils import stdout_off, stdout_on


def detect_dataset(
        model,
        dataset,
        out_file
):
    """

    :param model: MMdataParallel|dict dict(cfg_file, checkpoint)
    :param dataset: dict(cfg_file, img_dir, json_path)
    :return:
    """
    print("Detection.")
    # build model
    print("1.build model.", end=' ')
    stdout_off()
    if isinstance(model, dict):
        model = build_mmdet_model(model)
    else:
        print('skip.')
    assert isinstance(model, MMDataParallel)
    stdout_on()
    print("done.")

    # build data loader
    print("2.build data loader.", end=" ")
    stdout_off()
    data_loader = build_mmdet_dataset(**(dataset))

    if getattr(model.module, "CLASSES", None) is None:
        model.module.CLASSES = data_loader.dataset.CLASSES
    stdout_on()
    print('done.')

    # detect
    print("3.detect.", end='\n')
    stdout_off()
    distributed = False

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test( model=model,
                                   data_loader=data_loader,
                                   show=False,
                                   out_dir=None
                                   )
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)
    stdout_on()
    print('done.')

    # save result.
    print("4.save result.", end=' ')
    os.makedirs(os.path.split(out_file)[0], mode=0o777, exist_ok=True)
    if isinstance(outputs[0], list):
        js_data = data_loader.dataset._det2json(outputs)
    elif isinstance(outputs[0], tuple):
        js_data = data_loader.dataset._segm2json(outputs)[1]
    with open(out_file, "w") as f:
        json.dump(js_data, f, indent=4)
    print("done.")

    pass

