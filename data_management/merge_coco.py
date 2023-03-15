import json
from pathlib import Path
from datetime import datetime

REQUIRED_IMAGE_FIELDS = {
    "id",
    "file_name",
    "width",
    "height",
    "rights_holder",
    "datetime",
    "location",
    "corrupt",
    "seq_id",
    "seq_num_frames",
    "frame_num",
}

REQUIRED_ANNOTATION_FIELDS = {
    "id",
    "image_id",
    "category_id",
    "bbox",
    "sequence_level_annotation",
}


MERGED_PATH = Path("data/processed/qt-coyotes-merged.json")


COCO_FILES = [
    {
        "coco_path": "data/processed/CHIL/CHIL_uwin_mange_Marit_07242020.json",
        "data_path": "CHIL",
        "category_mapping": {
            1: 1,
            4: 1,
            2: 2,
            5: 2,
        },
    },
    {
        "coco_path": "data/processed/CHIL-earlier/CHIL_earlier.json",
        "data_path": "CHIL-earlier",
        "category_mapping": {
            1: 1,
            2: 2,
        },
    },
    {
        "coco_path": "data/processed/mange_images/mange_images.json",
        "data_path": "mange_images",
        "category_mapping": {
            1: 1,
            2: 2,
        },
    },
    {
        "coco_path": "data/processed/mange_Toronto/mange_Toronto.json",
        "data_path": "mange_Toronto",
        "category_mapping": {
            1: 1,
            2: 2,
        },
    },
]


def generate_info(coco_path: Path, years, descriptions, versions, contributors):
    try:
        with open(coco_path, "r") as f:
            coco = json.load(f)
            version = int(coco["info"]["version"]) + 1
    except FileNotFoundError:
        version = 1

    descriptions = list(
        f"{description} v{version} ({year})"
        for year, description, version in zip(years, descriptions, versions)
    )
    descriptions = ["qt coyotes merged dataset"] + descriptions
    contributors = ["qt coyotes"] + contributors

    info = {
        "version": str(version),
        "year": datetime.today().date().year,
        "description": ", ".join(descriptions),
        "contributor": ", ".join(contributors),
        "date_created": datetime.today().date().isoformat(),
    }
    return info


def main():
    merged_coco = {
        "categories": [
            {"id": 1, "name": "mange"},
            {"id": 2, "name": "no_mange"},
        ],
        "images": {},
        "annotations": {},
    }
    years = []
    descriptions = []
    versions = []
    contributors = []
    for coco_file in COCO_FILES:
        with open(coco_file["coco_path"]) as f:
            coco = json.load(f)
            years.append(coco["info"]["year"])
            descriptions.append(coco["info"]["description"])
            versions.append(coco["info"]["version"])
            contributors.append(coco["info"]["contributor"])
            data_path = Path(coco_file["data_path"])

            for image in coco["images"]:
                image_id = image["id"]
                if image_id in merged_coco["images"]:
                    raise ValueError(f"Duplicate image id {image_id}")
                for field in REQUIRED_IMAGE_FIELDS:
                    if field not in image:
                        raise ValueError(f"Missing field {field} in image {image_id}")
                image["file_name"] = str(data_path / image["file_name"])
                merged_coco["images"][image_id] = image

            for annotation in coco["annotations"]:
                annotation_id = annotation["id"]
                if annotation_id in merged_coco["annotations"]:
                    raise ValueError(f"Duplicate annotation id {annotation_id}")
                if annotation["category_id"] not in coco_file["category_mapping"]:
                    continue
                for field in REQUIRED_ANNOTATION_FIELDS:
                    if field not in annotation:
                        raise ValueError(
                            f"Missing field {field} in annotation {annotation_id}"
                        )
                annotation["category_id"] = coco_file["category_mapping"][
                    annotation["category_id"]
                ]
                merged_coco["annotations"][annotation_id] = annotation

    merged_coco["images"] = list(merged_coco["images"].values())
    merged_coco["annotations"] = list(merged_coco["annotations"].values())

    merged_coco["info"] = generate_info(
        MERGED_PATH, years, descriptions, versions, contributors
    )

    with open(MERGED_PATH, "w") as f:
        json.dump(merged_coco, f, indent=4)


if __name__ == "__main__":
    main()
