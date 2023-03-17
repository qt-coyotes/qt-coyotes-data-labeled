import json
import re
import uuid
import argparse
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

SEQ_RE1 = re.compile(r'([0-9]+):([0-9]+)\|([0-9]+)')
SEQ_RE2 = re.compile(r'([0-9]+)/([0-9]+)')
NUM_RE0 = re.compile(r'([0-9]+)\.[^\.]+$')


def image_source_row(row: pd.Series) -> Path:
    folder = Path(IMAGE_DATA_PATH)
    rel_path = row["RelativePath"].replace("\\", ".")

    return folder / f"{rel_path}.{row['File']}"


def get_category_id(row: pd.Series):
    return int(row["Disease"]) + 1


def generate_image(row: pd.Series):
    is_corrupt = False
    width = None
    height = None
    file_name = row["File"]

    photo_path = image_source_row(row)

    try:
        img = cv2.imread(str(photo_path))
        height, width, _ = img.shape
    except Exception as e:
        print(e)
        is_corrupt = True

    return {
        "id": str(uuid.uuid1()),
        "file_path": str(image_source_row(row).relative_to(ROOT_PATH)),
        "width": width,
        "height": height,
        "rights_holder": CONTRIBUTOR,
        "datetime": row["DateTime"].strip(),
        "location": f"{row['Location_Primary']}::{row['Location_Secondary']}",
        "location_park": row["Location_Primary"],
        "location_den": row["Location_Secondary"],
        "corrupt": is_corrupt,
    }


def generate_annotation(row: pd.Series, image: dict):
    annotations = {
        "id": str(uuid.uuid1()),
        "image_id": image["id"],
        "category_id": get_category_id(row),
        "bbox": [0, 0, image["width"], image["height"]],
        "width": image["width"],
        "height": image["height"],
        "num_adults": int(
            row["Count_Adult"] if not np.isnan(row["Count_Adult"]) else 0
        ),
        "num_individuals": 0,
    }

    for key in ["Count_Adult", "Count_Juvenile", "Count_UnkAge"]:
        if not np.isnan(row[key]):
            annotations["num_individuals"] += int(row[key])

    rename_map = {
        "DateTime": "date",
        "V_Nice": "nice_quality",
        "Location_Primary": "location_neighbourhood",
        "Location_Primary": "location_neighbourhood",
        "Location_Secondary": "location_den",
        "Cam_Brand": "camera_brand",
        "Species": "species",
        "Add_Sp": "additional_species",
        "Prey": "is_prey_visible",
        "Play": "is_playing",
        "Nursing": "is_nursing",
        "Resting": "is_resting",
        "Playing": "is_playing",
        "Eating": "is_eating",
        "Digging": "is_digging",
        "Colour": "is_colored_image",
        "Den_Int": "den_interaction",
        "Comments": "comments",
        "PlayObjType": "play_object_type",
        "PreyType": "prey_type",
        "Temperature": "temperature",
        "MoonPhase": "moon_phase",
        "CamModel": "camera_model",
        "Disease": "is_mange",
        "Count_Adult": "num_adult",
        "Count_Juvenile": "num_juvenile",
        "Count_UnkAge": "num_unknown_age",
        "Count_Female": "num_female",
        "Count_Male": "num_male",
        "Count_UnkSex": "num_unknown_sex",
    }

    for og, to in rename_map.items():
        v = row[og]
        if type(v) == np.float64 and np.isnan(row[og]):
            annotations[to] = None
        else:
            annotations[to] = row[og]

    return annotations


def generate_image_annotation(row: pd.Series):
    image = generate_image(row)
    annotation = generate_annotation(row, image)
    return image, annotation


def generate_image_sequences(df: pd.DataFrame, images: List[dict]):
    file_name_to_image = {image["file_path"]: image for image in images}

    seq_ids = {}
    nan_seq_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        if type(row["Sequence"]) == float and math.isnan(row["Sequence"]):
            ord_in_seq = float('nan')
            seq_length = float('nan')
            root_img_num = nan_seq_count
            nan_seq_count += 1
        elif (x := SEQ_RE1.fullmatch(row["Sequence"])):
            ord_in_seq = int(x[2])
            seq_length = int(x[3])
            root_img_num = int(x[1])
        elif (x := SEQ_RE2.fullmatch(row["Sequence"])):
            ord_in_seq = int(x[1])
            seq_length = int(x[2])
            img_num = int(NUM_RE0.search(row['File'])[1])
            root_img_num = img_num - ord_in_seq + 1
        else:
            raise Exception(f"No match for sequence id {row['Sequence']} in")

        if seq_ids.get(root_img_num) is None:
            seq_ids[root_img_num] = str(uuid.uuid1())
        seq_id = seq_ids[root_img_num]

        image = file_name_to_image[str(image_source_row(row))]
        image["seq_id"] = seq_id
        image["seq_num_frames"] = seq_length
        image["frame_num_in_seq"] = ord_in_seq

    return images


def generate_info(coco_path: Path):
    # This bit counts how many times this script was run
    try:
        with open(coco_path, "r") as f:
            coco = json.load(f)
            version = int(coco["info"]["version"]) + 1
    except FileNotFoundError:
        version = 1

    info = {
        "version": str(version),
        "year": 2020,
        "description": f"{DESCRIPTION}\n{EXTENDED_DESCRIPTION}",
        "contributor": CONTRIBUTOR,
        "date_created": datetime.today().date().isoformat(),
    }
    return info


def generate_categories():
    return [
        {"id": 1, "name": "coyote", "supercategory": "mange"},
        {"id": 2, "name": "coyote", "supercategory": "no_mange"},
    ]


def main():
    df = pd.read_csv(DATA_PATH)
    #df = df.drop_duplicates(subset=["File"])

    images = []
    annotations = []

    with ProcessPoolExecutor() as executor:
        futures = []
        for _, row in df.iterrows():
            future = executor.submit(generate_image_annotation, row)
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures)):
            image, annotation = future.result()
            images.append(image)
            annotations.append(annotation)

    images = generate_image_sequences(df, images)

    correct_len = len(df)

    assert len(images) == correct_len
    assert len(annotations) == correct_len

    coco = {
        "info": generate_info(COCO_PATH),
        "images": images,
        "categories": generate_categories(),
        "annotations": annotations,
    }

    with open(COCO_PATH, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('site_name', help="folder name")

    args = parser.parse_args()

    SITE_NAME = args.site_name
    ROOT_PATH = Path("raw_data")
    IMAGE_DATA_PATH = ROOT_PATH / SITE_NAME
    DATA_PATH = Path(f"csv_files/{SITE_NAME}.csv")
    COCO_PATH = Path(f"json_files/{SITE_NAME}.json")
    YEAR = 2022
    CONTRIBUTOR = "Sage Raymond <rraymon1@ualberta.ca>"
    DESCRIPTION = f"Coyote dens sites from {SITE_NAME} Edmonton {YEAR}"
    EXTENDED_DESCRIPTION = ""


    main()
