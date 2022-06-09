import json
import os
from tools.Image_preprocessing import get_image_info


def generate_test_json(
        img_file,
        ref_json,
        out_path,
):
    with open(ref_json, "r") as f:
        ref_data = json.load(f)
    js_data = dict(
        images=[],
        categories=ref_data['categories']
    )

    im_width, im_height, im_bands, im_geotrans, im_proj = get_image_info(img_file)
    img_name = os.path.split(img_file)[1]
    img_info = dict(
        height = im_height,
        width = im_width,
        id = 0,
        file_name = img_name
    )
    js_data["images"].append(img_info)

    # save json
    out_json = os.path.join(out_path, "test.json")
    os.makedirs(out_path, mode=0o777, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(js_data, f, sort_keys=False, indent=4)

