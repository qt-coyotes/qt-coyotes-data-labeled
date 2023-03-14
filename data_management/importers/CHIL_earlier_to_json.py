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

RAW_DATA_PATH = Path("data/raw/CHIL-earlier")
# There are 3 very similar metadata files, this is the correct one according to
# the image file names, and that "Mange photo data June 1 2018.xlsx" is a subset
# of this one:
XLSX_PATH = (
    RAW_DATA_PATH / "Data spreadsheets/Mange photo data June 1 2018 CC(1).xlsx"
)
COCO_PATH = Path("data/processed/CHIL-earlier/CHIL_earlier.json")
CONTRIBUTOR = "Maureen Murray"
EXTENDED_DESCRIPTION = """ID	ID number in database output
path	Full file name path
species	Based on data scraping from database, should all be coyote
date	Date photo was taken, extracted from database output
time	Time photo was taken, extracted from database output
new_file_name	Systematic file name containing the site ID and season
File_name_order	Order in which photos are numbered in file name
Mange_signs_present	1 indicates that photo contains at least one coyote exhibiting signs of mange including hair loss and/or lesions on the tail, legs, body, or face
Mange_potential_low_confidence	1 indicates that coyote exhibits potential signs of mange but confidence is low because of angle, photo quality, ambiguous signs, etc.
Clear photo	1 indicates that the coyote in the photo is mostly in the field of view and not too blurry, far away, dark, or bright to detect mange signs if they were present
Blurry	1 indicates that coyote was too blurry in the photograph to assess mange signs
Out_of_frame	1 indicates that at least 50% of the coyote was out of view but no signs were seen in the visible part of the body
Lighting	1 indicates that mange signs could not be accurately assessed because the photo was overexposed or too dark to see changes in coat
Too_far_away	1 indicates that coyote was too small in the photograph to confidently assess coat quality
Not_coyote	1 indicates that animal in the photograph was not a coyote
In_color	1 indicates that the photo was in color, which likely improves the detectability of mange
Whole_body	1 indicates that the coyote's whole body (at least on side of the tail, haunches, legs, flanks, neck, and face) was visible, increasing confidence in true negatives
Multiple_coyotes	1 indicates that at least 2 coyotes were in the photograph. Mange might be less likely for coyotes in packs, but packs with at least one mangy member are more likely to transmit the parasite
Notes	Description of mange for all positive cases
Inspected	Indicator variable that photo was looked at
Season	Calendar season in which the photo was taken
Year	Year in which the photo was taken
Site	Three letter code of the site at which the photo was taken"""
MAX_SEQUENCE_DELTA = timedelta(minutes=30)
WIDTH_HEIGHT_TO_BAR_HEIGHT = {(2592, 1944): 200, (3264, 2448): 100}
DUPLICATE_FILE_NAMES = set([])


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
    file_name = f'{row["Season"]} {row["Year"]}/{row["new_file_name"]}'
    photo_path = RAW_DATA_PATH / file_name
    try:
        img = cv2.imread(str(photo_path))
        if img is None:
            corrupt = True
        height, width, _ = img.shape
    except Exception as e:
        print(e)
        corrupt = True

    if not pd.isnull(row["date"]):
        dt: datetime = pd.to_datetime(row["date"])
        if not pd.isnull(row["time"]) and not pd.isna(row["time"]):
            td = pd.to_timedelta(str(row['time']))
            dt += td
            datetime_str = dt.isoformat(sep=" ", timespec="seconds")
        else:
            datetime_str = dt.date().isoformat()
    elif not corrupt:
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
                    f"Warning: could not parse datetime in {file_name}: "
                    + raw_text
                )
                corrupt = True
    else:
        datetime_str = None

    if datetime_str is not None and len(datetime_str) == 0:
        datetime_str = None

    return {
        "id": str(uuid.uuid1()),
        "file_name": f'{row["Season"]} {row["Year"]}/{row["new_file_name"]}',
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

    if width is None or height is None:
        bbox = None
    else:
        barheight = WIDTH_HEIGHT_TO_BAR_HEIGHT.get((width, height))
        if barheight is None:
            raise Exception(f"Unknown width, height: {width}, {height}")
        else:
            bbox = [0, 0, width, height - barheight]

    return {
        "id": str(uuid.uuid1()),
        "image_id": image["id"],
        "category_id": category_id,
        "bbox": bbox,
        "sequence_level_annotation": False,
        "mange_potential_low_confidence": int(
            row["Mange_potential_low_confidence"]
        )
        if not pd.isna(row["Mange_potential_low_confidence"])
        else 0,
        "clear_photo": int(row["Clear photo"])
        if not pd.isna(row["Clear photo"])
        else 0,
        "blurry": int(row["Blurry"]) if not pd.isna(row["Blurry"]) else 0,
        "out_of_frame": int(row["Out_of_frame"])
        if not pd.isna(row["Out_of_frame"])
        else 0,
        "lighting": int(row["Lighting"]) if not pd.isna(row["Lighting"]) else 0,
        "too_far_away": int(row["Too_far_away"])
        if not pd.isna(row["Too_far_away"])
        else 0,
        "in_color": int(row["In_color"]) if not pd.isna(row["In_color"]) else 0,
        "whole_body": int(row["Whole_body"])
        if not pd.isna(row["Whole_body"])
        else 0,
        "multiple_coyotes": int(row["Multiple_coyotes"])
        if not pd.isna(row["Multiple_coyotes"])
        else 0,
        "notes": row["Notes"] if not pd.isna(row["Notes"]) else None,
        "inspected": int(row["Inspected"])
        if not pd.isna(row["Inspected"])
        else 0,
        "season": row["Season"].strip() if not pd.isna(row["Season"]) else None,
        "year": int(row["Year"]) if not pd.isna(row["Year"]) else None,
        "percent_body_visible": float(row["Percent body visible"])
        if not pd.isna(row["Percent body visible"])
        else None,
        "percent_visible_body_affected": float(
            row["Percent visible body affected"]
        )
        if not pd.isna(row["Percent visible body affected"])
        else None,
        "severity_of_affected_area": row["Severity of affected area (mild, moderate, severe)"]
        if not pd.isna(row["Severity of affected area (mild, moderate, severe)"])
        else None,
        "confidence": row["Confidence (low, med, high)"]
        if not pd.isna(row["Confidence (low, med, high)"])
        else None,
    }


def generate_image_annotation(row: pd.Series):
    image = generate_image(row)
    annotation = generate_annotation(row, image)
    return image, annotation


def generate_image_sequences(df: pd.DataFrame, images: List[dict]):
    file_name_to_image = {image["file_name"]: image for image in images}

    df = df.sort_values(by=["Site", "Season", "Year", "new_file_name"])

    prev_location_name = None
    prev_dt = None
    seq_ids = set()
    sequence = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        file_name = f'{row["Season"]} {row["Year"]}/{row["new_file_name"]}'
        image = file_name_to_image[file_name]
        location_name = row["Site"]
        if image["datetime"] is not None:
            try:
                dt = datetime.strptime(image["datetime"], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                try:
                    dt = datetime.strptime(image["datetime"], "%Y-%m-%d %H:%M")
                except ValueError:
                    dt = datetime.strptime(image["datetime"], "%Y-%m-%d")
        if prev_location_name is not None:
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
        {"id": 4, "name": "fox", "supercategory": "mange"},
        {"id": 5, "name": "fox", "supercategory": "no_mange"},
        {"id": 6, "name": "fox", "supercategory": "unknown"},
    ]


def main():
    df = pd.read_excel(XLSX_PATH)

    df = df.sort_values(
        by=["new_file_name", "Mange_signs_present"], ascending=[True, False]
    )
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

    COCO_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(COCO_PATH, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=4)


if __name__ == "__main__":
    main()
