import os
import json
from PIL import Image
from tqdm import tqdm

table = {
    "PC900 PROFESSIONAL::2048x1536": [34, 1470],
    "PC800 PROFESSIONAL::2048x1536": [34, 1470],
    "BTC-7E-H4::4608x2592": [0, 2360],
    "BTC-7E-H4::1920x1080": [0, 982],
    "nan::1920x1080": [0, 982],
}

def jprint(j):
    print(json.dumps(j, indent=4))

def tprint(t):
    print(f"==== {t} ====")

def lprint(l):
    print('[')
    for x in l:
        print("    " + x)
    print(']')

def ann2key(ann) -> str:
    c = ann['camera_model']
    dims = f"{ann['width']}x{ann['height']}"
    return f"{c}::{dims}"


for fname in filter(lambda x: x.endswith('.json'), os.listdir('json_files')):
    tprint(fname[:-5])
    full_path = f"json_files/{fname}"
    raw_path = f"raw_images/{fname[:-5]}"
    crop_path = f"cropped_images/{fname[:-5]}"

    with open(full_path, 'r') as f:
        df = json.load(f)

    for ann in tqdm(df['annotations']):
        key = ann2key(ann)
        image_id = ann['image_id']

        image_path = None
        for im in df['images']:
            if im['id'] == image_id:
                image_path = im['file_path']
                break;

        if image_path is None:
            print(image_id)
            jprint(ann)
            exit(1)

        image = Image.open(f"raw_images/{image_path}")
        w, h = image.size

        u, l = table[ann2key(ann)]
        image.crop((0, u, w, l)).save(f"cropped_images/{image_path}")


"""
    for j in js['annotations']:
        assert table.get(ann2key(j)) is not None

        c = j['camera_model']
        dims = f"{j['width']}x{j['height']}"
        hood = f"{j['location_neighbourhood']}::{j['location_den']}"


tprint("Unique camera models")
jprint(camera_models)
tprint("Unlabled hoods")
print(unlabled_pics)
tprint("Errors")
lprint(errors)
#jprint(js)
"""
