import json
import os


def merge_category(
        result_json,
):
    with open(result_json, 'r') as f:
        res = json.load(f)
    for r in res:
        r["category_id"] = 0

    save_path = os.path.splitext(result_json)[0] + "_onecat.json"
    with open(save_path, "w") as f:
        json.dump(res, f)

    return save_path
