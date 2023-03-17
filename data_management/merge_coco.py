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
    {
        "coco_path": "data/processed/coyote-dens/CumberlandA.json",
        "data_path": "coyote-dens",
        "category_mapping": {
            1: 1,
            2: 2,
        },
    },
    {
        "coco_path": "data/processed/coyote-dens/CumberlandB.json",
        "data_path": "coyote-dens",
        "category_mapping": {
            1: 1,
            2: 2,
        },
    },
    {
        "coco_path": "data/processed/coyote-dens/FalconerA.json",
        "data_path": "coyote-dens",
        "category_mapping": {
            1: 1,
            2: 2,
        },
    },
    {
        "coco_path": "data/processed/coyote-dens/FalconerB.json",
        "data_path": "coyote-dens",
        "category_mapping": {
            1: 1,
            2: 2,
        },
    },
    {
        "coco_path": "data/processed/coyote-dens/KinnardA.json",
        "data_path": "coyote-dens",
        "category_mapping": {
            1: 1,
            2: 2,
        },
    },
    {
        "coco_path": "data/processed/coyote-dens/KinnardB.json",
        "data_path": "coyote-dens",
        "category_mapping": {
            1: 1,
            2: 2,
        },
    },
    {
        "coco_path": "data/processed/coyote-dens/KinnardC.json",
        "data_path": "coyote-dens",
        "category_mapping": {
            1: 1,
            2: 2,
        },
    },
    {
        "coco_path": "data/processed/coyote-dens/RowlandC.json",
        "data_path": "coyote-dens",
        "category_mapping": {
            1: 1,
            2: 2,
        },
    },
    {
        "coco_path": "data/processed/coyote-dens/RowlandE.json",
        "data_path": "coyote-dens",
        "category_mapping": {
            1: 1,
            2: 2,
        },
    },
    {
        "coco_path": "data/processed/coyote-dens/RowlandF.json",
        "data_path": "coyote-dens",
        "category_mapping": {
            1: 1,
            2: 2,
        },
    },
    {
        "coco_path": "data/processed/coyote-dens/RowlandH.json",
        "data_path": "coyote-dens",
        "category_mapping": {
            1: 1,
            2: 2,
        },
    },
    {
        "coco_path": "data/processed/coyote-dens/RowlandJ.json",
        "data_path": "coyote-dens",
        "category_mapping": {
            1: 1,
            2: 2,
        },
    },
    {
        "coco_path": "data/processed/coyote-dens/RowlandK.json",
        "data_path": "coyote-dens",
        "category_mapping": {
            1: 1,
            2: 2,
        },
    },
    {
        "coco_path": "data/processed/coyote-dens/RowlandL.json",
        "data_path": "coyote-dens",
        "category_mapping": {
            1: 1,
            2: 2,
        },
    },
    {
        "coco_path": "data/processed/coyote-dens/RowlandN.json",
        "data_path": "coyote-dens",
        "category_mapping": {
            1: 1,
            2: 2,
        },
    },
    {
        "coco_path": "data/processed/coyote-dens/StrathearnA.json",
        "data_path": "coyote-dens",
        "category_mapping": {
            1: 1,
            2: 2,
        },
    },
    {
        "coco_path": "data/processed/coyote-dens/StrathearnB.json",
        "data_path": "coyote-dens",
        "category_mapping": {
            1: 1,
            2: 2,
        },
    },
    {
        "coco_path": "data/processed/coyote-dens/WagnerB.json",
        "data_path": "coyote-dens",
        "category_mapping": {
            1: 1,
            2: 2,
        },
    },
    {
        "coco_path": "data/processed/coyote-dens/WagnerC.json",
        "data_path": "coyote-dens",
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
        "info": None,
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

            image_id_to_image = {image["id"]: image for image in coco["images"]}

            for image in coco["images"]:
                image_id = image["id"]
                if image["corrupt"]:
                    continue
                if image_id in merged_coco["images"]:
                    raise ValueError(f"Duplicate image id {image_id}")
                for field in REQUIRED_IMAGE_FIELDS:
                    if field not in image:
                        print(
                            f"Warning Missing field {field} in image {image_id} in {coco_file['coco_path']}"
                        )
                        image[field] = None

                # hack: coyote-dens missing required field file_name
                if image["file_name"] is None:
                    image["file_name"] = image["file_path"]
                image["file_name"] = str(data_path / image["file_name"])
                merged_coco["images"][image_id] = image

            for annotation in coco["annotations"]:
                annotation_id = annotation["id"]
                if image_id_to_image[annotation["image_id"]]["corrupt"]:
                    continue
                if annotation_id in merged_coco["annotations"]:
                    raise ValueError(f"Duplicate annotation id {annotation_id}")
                if (
                    annotation["category_id"]
                    not in coco_file["category_mapping"]
                ):
                    continue
                for field in REQUIRED_ANNOTATION_FIELDS:
                    if field not in annotation:
                        print(
                            f"Missing field {field} in annotation {annotation_id} in {coco_file['coco_path']}"
                        )
                        annotation[field] = None
                if coco_file["data_path"] == "coyote-dens":
                    annotation["category_id"] = (
                        1 if annotation["is_mange"] else 2
                    )
                else:
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
