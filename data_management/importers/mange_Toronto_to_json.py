import json
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import cv2
import pandas as pd
from tqdm import tqdm

RAW_DATA_PATH = Path("data/raw/mange_Toronto")
CSV_PATH = RAW_DATA_PATH / "coyotes_mange_data_19072022.csv"
COCO_PATH = Path("data/processed/mange_Toronto/mange_Toronto.json")
CONTRIBUTOR = "Tiziana Gelmi Candusso"
EXTENDED_DESCRIPTION = """"""
MAX_SEQUENCE_DELTA = timedelta(minutes=30)
WIDTH_HEIGHT_TO_BAR_HEIGHT = {
    (1024, 768): 62,
    (1920, 1440): 96,
    (4416, 3312): 197,
    (4608, 3456): 144,
    (4624, 3468): 256
}

DUPLICATE_FILE_NAMES = set()


def get_category_id(row: pd.Series):
    if row["Species"] != "coyote":
        category_id = 4
    elif row["mange"] is True:
        category_id = 1
    elif row["mange"] is False:
        category_id = 2
    elif pd.isna(row["mange"]):
        category_id = 3
    else:
        raise Exception(
            f"Unknown mange: {row['mange']}"
        )
    return category_id


def generate_image(row: pd.Series):
    corrupt = False
    width = None
    height = None
    file_name = row["RelativePath"].replace('\\', '.') + "." + row["File"]
    photo_path = RAW_DATA_PATH / file_name
    try:
        img = cv2.imread(str(photo_path))
        height, width, _ = img.shape
    except Exception as e:
        print(e)
        corrupt = True

    dt = pd.to_datetime(row["DateTime"])
    datetime_str = dt.isoformat(sep=" ", timespec="seconds")

    return {
        "id": str(uuid.uuid1()),
        "file_name": file_name,
        "width": width,
        "height": height,
        "rights_holder": CONTRIBUTOR,
        "datetime": datetime_str,
        "location": row["RelativePath"],
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

        "image_quality": row["ImageQuality"] if not pd.isna(row["ImageQuality"]) else None,
        "delete_flag": row["DeleteFlag"] if not pd.isna(row["DeleteFlag"]) else None,
        "n_species": float(row["Nspecies"]) if not pd.isna(row["Nspecies"]) else None,
        "leash": row["leash"] if not pd.isna(row["leash"]) else None,
        "melanistic": row["melanistic"] if not pd.isna(row["melanistic"]) else None,
        "species_extra": row["SpeciesExtra"]
        if not pd.isna(row["SpeciesExtra"])
        else None,
        "UWIN": row["UWIN"]
        if not pd.isna(row["UWIN"])
        else None,
        "spatiotemp_proj": row["spatiotemp_proj"]
        if not pd.isna(row["spatiotemp_proj"])
        else None,
        "conn_val": row["Conn_val"]
        if not pd.isna(row["Conn_val"])
        else None,
        "speciesextraname": row["speciesextraname"]
        if not pd.isna(row["speciesextraname"])
        else None,
        "revised": row["revised"] if not pd.isna(row["revised"]) else None,
        "flagged": row["flagged"] if not pd.isna(row["flagged"]) else None,
        "vehicles": row["vehicles"] if not pd.isna(row["vehicles"]) else None
    }


def generate_image_annotation(row: pd.Series):
    image = generate_image(row)
    annotation = generate_annotation(row, image)
    return image, annotation


def generate_image_sequences(df: pd.DataFrame, images: List[dict]):
    file_name_to_image = {image["file_name"]: image for image in images}

    df = df.sort_values(by=["RelativePath", "File"])

    prev_location_name = None
    prev_dt = None
    seq_ids = set()
    sequence = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        file_name = row["RelativePath"].replace('\\', '.') + "." + row["File"]
        image = file_name_to_image[file_name]
        location_name = row["RelativePath"]
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
        "description": "Toronto Urban Wildlife Information Network Mange Dataset\n"
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

    df = df.sort_values(by=["RelativePath", "File", "mange"], ascending=[True, True, False])
    df = df.drop_duplicates(subset=["RelativePath", "File"])

    df = df[~df["RelativePath"].isin(DUPLICATE_FILE_NAMES)]

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
