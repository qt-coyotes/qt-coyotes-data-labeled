import cv2
import glob
import json
import os
from tqdm import tqdm

DUPLICATE_JSON_PATH = "data/processed/CHIL/duplicate_images.json"
GLOB_PATTERNS = [
    "data/raw/CHIL/**/*.jpg",
    "data/raw/CHIL/**/*.JPG",
    "data/raw/CHIL_earlier/**/*.jpg",
    "data/raw/CHIL_earlier/**/*.JPG",
    "data/raw/mange_images/**/*.JPG",
    "data/raw/mange_images/**/*.jpg"
]


def identify_duplicate_images(filenames):
    os.environ["PYTHONHASHSEED"] = "0"
    hash_filenames = {}
    for filename in tqdm(filenames):
        image = cv2.imread(filename)
        h = hash(image.data.tobytes())
        if h in hash_filenames:
            hash_filenames[h].append(filename)
        else:
            hash_filenames[h] = [filename]
    duplicates = sorted(filter(
        lambda filenames: len(filenames) > 1, hash_filenames.values()
    ))
    return duplicates


def main():
    filenames = []
    for glob_pattern in GLOB_PATTERNS:
        filenames.extend(glob.glob(glob_pattern, recursive=True))
    duplicates = identify_duplicate_images(filenames)
    with open(DUPLICATE_JSON_PATH, "w") as f:
        json.dump(duplicates, f, indent=4)
    print(f"Found {len(duplicates)} duplicate images.")
    print(f"See: {DUPLICATE_JSON_PATH}")
    assert len(duplicates) == 0


if __name__ == "__main__":
    main()
