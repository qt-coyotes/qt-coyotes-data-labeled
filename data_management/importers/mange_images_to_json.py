import json
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import cv2
import pandas as pd
from tqdm import tqdm

RAW_DATA_PATH = Path("data/raw/mange_images")
CSV_PATH = RAW_DATA_PATH / "mange_metadata.csv"
COCO_PATH = Path("data/processed/mange_images/mange_images.json")
CONTRIBUTOR = "Mason Fidino"
EXTENDED_DESCRIPTION = """"""
MAX_SEQUENCE_DELTA = timedelta(minutes=30)

DUPLICATE_FILE_NAMES = set()


def get_category_id(row: pd.Series):
    if row["Not_coyote"] == 1:
        category_id = 4
    elif row["Mange_signs_present"] == 1:
        category_id = 1
    elif row["Mange_signs_present"] == 0:
        category_id = 2
    elif pd.isna(row["Mange_signs_present"]):
        category_id = 3
    else:
        raise Exception(
            f"Unknown Mange_signs_present: {row['Mange_signs_present']}"
        )
    return category_id


def generate_image(row: pd.Series):
    corrupt = False
    width = None
    height = None
    file_name = row["new_file_name"]
    photo_path = RAW_DATA_PATH / file_name
    try:
        img = cv2.imread(str(photo_path))
        height, width, _ = img.shape
    except Exception as e:
        print(e)
        corrupt = True

    td = pd.to_timedelta(row["time"])
    dt = pd.to_datetime(row["date"])
    dt += td
    datetime_str = dt.isoformat(sep=" ", timespec="seconds")

    return {
        "id": str(uuid.uuid1()),
        "file_name": row["new_file_name"],
        "width": width,
        "height": height,
        "rights_holder": CONTRIBUTOR,
        "datetime": datetime_str,
        "location": row["Site"],
        "corrupt": corrupt,
    }


def generate_annotation(row: pd.Series, image: dict):
    category_id = get_category_id(row)

    width, height = image["width"], image["height"]

    return {
        "id": str(uuid.uuid1()),
        "image_id": image["id"],
        "category_id": category_id,
        "bbox": [0, 0, width, height - 198],
        "sequence_level_annotation": False,

        "blur_match": row["blur.match"] if not pd.isna(row["blur.match"]) else None,
        "blur_match_1": row["blur.match.1"] if not pd.isna(row["blur.match.1"]) else None,
        "blur": float(row["blur.match.1"]) if not pd.isna(row["blur.match.1"]) else None,
        "mange_potential_low_confidence": int(row["Mange_potential_low_confidence"])
        if not pd.isna(row["Mange_potential_low_confidence"]) else 0,
        "clear_photo": int(row["Clear.photo"])
        if not pd.isna(row["Clear.photo"]) else None,
        "blurry": int(row["Blurry"])
        if not pd.isna(row["Blurry"])
        else 0,
        "lighting": int(row["Lighting"])
        if not pd.isna(row["Lighting"])
        else 0,
        "too_far_away": int(row["Too_far_away"])
        if not pd.isna(row["Too_far_away"])
        else 0,
        "in_color": int(row["In_color"])
        if not pd.isna(row["In_color"])
        else None,
        "whole_body": int(row["Whole_body"])
        if not pd.isna(row["Whole_body"])
        else 0,
        "multiple_coyotes": int(row["Multiple_coyotes"])
        if not pd.isna(row["Multiple_coyotes"])
        else 0,
        "notes": row["Notes"] if not pd.isna(row["Notes"]) else None,
        "inspected": int(row["Inspected"]) if not pd.isna(row["Inspected"]) else 0,
        "season": row["Season"] if not pd.isna(row["Season"]) else None,
        "year": int(row["Year"]) if not pd.isna(row["Year"]) else None,
        "propbodyvis": int(row["propbodyvis"])
        if not pd.isna(row["propbodyvis"])
        else None,
        "propbodyvismange": float(row["propbodyvismange"])
        if not pd.isna(row["propbodyvismange"])
        else None,
        "sevaffarea": row["sevaffarea"] if not pd.isna(row["sevaffarea"]) else None,
        "carterconfidence": row["carterconfidence"]
        if not pd.isna(row["carterconfidence"]) else None,
        "overallseverity": row["overallseverity"]
        if not pd.isna(row["overallseverity"]) else None,
        "maureenconfidence": row["maureenconfidence"]
        if not pd.isna(row["maureenconfidence"]) else None
    }


def generate_image_annotation(row: pd.Series):
    image = generate_image(row)
    annotation = generate_annotation(row, image)
    return image, annotation


def generate_image_sequences(df: pd.DataFrame, images: List[dict]):
    file_name_to_image = {image["file_name"]: image for image in images}

    df = df.sort_values(by=["Site", "new_file_name"])

    prev_location_name = None
    prev_dt = None
    seq_ids = set()
    sequence = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        file_name = row["new_file_name"]
        image = file_name_to_image[file_name]
        location_name = row["Site"]
        try:
            dt = datetime.strptime(image["datetime"], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            dt = datetime.strptime(image["datetime"], "%Y-%m-%d %H:%M")
        if prev_location_name is not None and prev_dt is not None:
            delta = dt - prev_dt
            if (
                prev_location_name != location_name
                or delta > MAX_SEQUENCE_DELTA
            ):
                while True:
                    seq_id = str(uuid.uuid1())
                    if seq_id not in seq_ids:
                        seq_ids.add(seq_id)
                        break
                for frame_num, sequence_image in enumerate(sequence):
                    sequence_image["seq_id"] = seq_id
                    sequence_image["seq_num_frames"] = len(sequence)
                    sequence_image["frame_num"] = frame_num
                sequence = []
        sequence.append(image)
        prev_location_name = location_name
        prev_dt = dt

    # flush last sequence
    while True:
        seq_id = str(uuid.uuid1())
        if seq_id not in seq_ids:
            seq_ids.add(seq_id)
            break
    for frame_num, image in enumerate(sequence):
        image["seq_id"] = seq_id
        image["seq_num_frames"] = len(sequence)
        image["frame_num"] = frame_num
    return images


def generate_info(coco_path: Path):
    try:
        with open(coco_path, "r") as f:
            coco = json.load(f)
            version = int(coco["info"]["version"]) + 1
    except FileNotFoundError:
        version = 1

    info = {
        "version": str(version),
        "year": 2013,
        "description": "Chicago Urban Wildlife Information Network Mange Dataset\n"
        + EXTENDED_DESCRIPTION,
        "contributor": CONTRIBUTOR,
        "date_created": datetime.today().date().isoformat(),
    }
    return info


def generate_categories():
    return [
        {"id": 1, "name": "coyote", "supercategory": "mange"},
        {"id": 2, "name": "coyote", "supercategory": "no_mange"},
        {"id": 3, "name": "coyote", "supercategory": "unknown"},
        {"id": 4, "name": "not coyote", "supercategory": "unknown"},
    ]


def main():
    df = pd.read_csv(CSV_PATH)

    df = df.sort_values(by=["new_file_name", "Mange_signs_present"], ascending=[True, False])
    df = df.drop_duplicates(subset=["new_file_name"])

    df = df[~df["new_file_name"].isin(DUPLICATE_FILE_NAMES)]

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
    main()
