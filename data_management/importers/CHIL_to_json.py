import json
import re
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd
import pytesseract
from tqdm import tqdm

RAW_DATA_PATH = Path("data/raw/CHIL")
XLSX_PATH = RAW_DATA_PATH / "CHIL_uwin_mange_Marit_07242020.xlsx"
COCO_PATH = Path("data/processed/CHIL/CHIL_uwin_mange_Marit_07242020.json")
CONTRIBUTOR = "Maureen Murray"
EXTENDED_DESCRIPTION = """updateCommonName	Confirm species ID
updateNumIndividuals	Confirm number of individuals in photo
File_name_order	consecutive numbering
Inspected	1 indicates that this photo was checked
Mange_signs_present	1 indicates that photo contains at least one coyote exhibiting signs of mange including hair loss and/or lesions on the tail, legs, body, or face
In_color	1 indicates that the photo was in color, which likely improves the detectability of mange
Season	Calendar season in which the photo was taken (Spring, Summer, Fall, Winter)
Year	Year in which the photo was taken
propbodyvis	Proportion of the coyote's body that is visible in the photo, expressed from 0 - 1. For example, if a coyote is in perfect profile from nose to tail it would get a value of 0.5
propbodyvismange	Proportion of the coyote's body that is visible and affected by mange
severity	Mild < 25%, Moderate = 25 - 50%, Severe > 50%
confidence	High, medium, low - subjective!
flagged for follow up	1 indicates that coyote exhibits potential signs of mange but confidence is low because of angle, photo quality, ambiguous signs, etc."""
MAX_SEQUENCE_DELTA = timedelta(minutes=30)

DUPLICATE_FILE_NAMES = set([
    "VID3480-00002.jpg",
    "VID3480-00012.jpg",
    "VID3480-00020.jpg",
    "VID3480-00031.jpg",
    "VID3480-00038.jpg",
    "VID3480-00040.jpg",
    "VID3480-00041.jpg",
    "VID3480-00042.jpg",
    "VID3480-00043.jpg",
    "VID3480-00044.jpg",
    "VID3480-00045.jpg",
    "VID3480-00046.jpg",
    "VID3480-00047.jpg",
    "VID3480-00052.jpg",
    "VID3480-00034.jpg",
    "VID3480-00036.jpg",
    "VID3480-00037.jpg",
    "VID3480-00048.jpg",
    "VID5017-00099.jpg"
])


def get_category_id(row: pd.Series):
    commonName = (
        row["updateCommonName"]
        if not pd.isna(row["updateCommonName"])
        else row["commonName"]
    )
    if commonName == "Coyote":
        if row["Mange_signs_present"] == 1:
            category_id = 1
        elif row["Mange_signs_present"] == 0:
            category_id = 2
        elif pd.isna(row["Mange_signs_present"]):
            category_id = 3
        else:
            raise Exception(
                f"Unknown Mange_signs_present: {row['Mange_signs_present']}"
            )
    elif commonName == "Red fox":
        if row["Mange_signs_present"] == 1:
            category_id = 4
        elif row["Mange_signs_present"] == 0:
            category_id = 5
        elif pd.isna(row["Mange_signs_present"]):
            category_id = 6
        else:
            raise Exception(
                f"Unknown Mange_signs_present: {row['Mange_signs_present']}"
            )
    else:
        print(row)
        raise Exception(f"Unknown commonName: {commonName}")
    return category_id


def generate_image(row: pd.Series):
    corrupt = False
    width = None
    height = None
    file_name = row["photoName"]
    photo_path = RAW_DATA_PATH / file_name
    try:
        img = cv2.imread(str(photo_path))
        height, width, _ = img.shape
    except Exception as e:
        print(e)
        corrupt = True

    # https://stackoverflow.com/questions/62560122/applying-user-patterns-in-pytesseract
    img_right_border = img[:, -1, :]
    bar_height = 0
    for i in reversed(range(img_right_border.shape[0])):
        if np.all(img_right_border[i, :] >= 253):
            bar_height += 1
        else:
            break
    if bar_height == 0:
        print(f"Warning: could not find bar in {file_name}")
        corrupt = True
        bar_height = 100
    raw_text = pytesseract.image_to_string(
        img[-bar_height:, -10 * bar_height:, :],
        config='-c tessedit_char_whitelist="0123456789-: " --psm 7',
    )
    text = raw_text.strip()
    text = re.sub(r" +:", ":", text)
    text = re.sub(r": +", ":", text)
    datetime_str = text
    try:
        dt = datetime.strptime(text, "%m-%d-%Y %H:%M:%S")
        datetime_str = dt.isoformat(sep=" ", timespec="seconds")
    except Exception:
        try:
            dt = datetime.strptime(text, "%m-%d-%Y %H:%M")
            datetime_str = dt.isoformat(sep=" ", timespec="minutes")
        except Exception:
            print(
                f"Warning: could not parse datetime in {file_name}: " + raw_text
            )
            corrupt = True

    return {
        "id": str(uuid.uuid1()),
        "file_name": row["photoName"],
        "width": width,
        "height": height,
        "rights_holder": CONTRIBUTOR,
        "datetime": datetime_str,
        "location": row["locationName"],
        "corrupt": corrupt,
    }


def generate_annotation(row: pd.Series, image: dict):
    category_id = get_category_id(row)

    width, height = image["width"], image["height"]

    numIndividuals = (
        int(row["updateNumIndividuals"])
        if not pd.isna(row["updateNumIndividuals"])
        else int(row["numIndividuals"])
    )

    # some notes are misplaced in the good picture quality column
    good_picture_quality = None
    try:
        good_picture_quality = (
            int(row["good picture quality"])
            if not pd.isna(row["good picture quality"])
            else None
        )
    except ValueError:
        if pd.isna(row["Notes"]):
            row["Notes"] = row["good picture quality"]

    return {
        "id": str(uuid.uuid1()),
        "image_id": image["id"],
        "category_id": category_id,
        "bbox": [0, 0, width, height - 198],
        "sequence_level_annotation": False,
        "city": row["City"] if not pd.isna(row["City"]) else None,
        "inspected": int(row["Inspected"])
        if not pd.isna(row["Inspected"])
        else 0,
        "num_individuals": numIndividuals,
        "in_color": int(row["In_color"])
        if not pd.isna(row["In_color"])
        else None,
        "season": row["Season"] if not pd.isna(row["Season"]) else None,
        "year": int(row["Year"]) if not pd.isna(row["Year"]) else None,
        "propbodyvis": row["propbodyvis"]
        if not pd.isna(row["propbodyvis"])
        else None,
        "propbodyvismange": row["propbodyvismange"]
        if not pd.isna(row["propbodyvismange"])
        else None,
        "severity": row["severity"] if not pd.isna(row["severity"]) else None,
        "confidence": row["confidence"]
        if not pd.isna(row["confidence"])
        else None,
        "flagged_for_follow_up": int(row["flagged for follow up"])
        if not pd.isna(row["flagged for follow up"])
        else None,
        "good_picture_quality": good_picture_quality,
        "notes": row["Notes"] if not pd.isna(row["Notes"]) else None,
    }


def generate_image_annotation(row: pd.Series):
    image = generate_image(row)
    annotation = generate_annotation(row, image)
    return image, annotation


def generate_image_sequences(df: pd.DataFrame, images: List[dict]):
    file_name_to_image = {image["file_name"]: image for image in images}

    df = df.sort_values(by=["locationName", "photoName"])

    prev_location_name = None
    prev_dt = None
    seq_ids = set()
    sequence = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        file_name = row["photoName"]
        image = file_name_to_image[file_name]
        location_name = row["locationName"]
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
        "year": 2020,
        "description": "Chicago Urban Urban Wildlife Information Network Mange Dataset\n"
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
        {"id": 4, "name": "fox", "supercategory": "mange"},
        {"id": 5, "name": "fox", "supercategory": "no_mange"},
        {"id": 6, "name": "fox", "supercategory": "unknown"},
    ]


def main():
    df = pd.read_excel(XLSX_PATH)

    df = df.sort_values(by=["photoName", "Mange_signs_present"], ascending=[True, False])
    df = df.drop_duplicates(subset=["photoName"])

    df = df[~df["photoName"].isin(DUPLICATE_FILE_NAMES)]

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
