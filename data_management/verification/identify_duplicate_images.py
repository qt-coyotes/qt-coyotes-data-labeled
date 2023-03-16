import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
from tqdm import tqdm

DUPLICATE_JSON_PATH = "data/processed/duplicate_images.json"
GLOB_PATTERNS = [
    "data/raw/**/*.jpg",
    "data/raw/**/*.JPG"
]
NUMBER_OF_THREADS = 64


def get_image_hash(filename):
    image = cv2.imread(filename)
    h = hash(image.data.tobytes())
    return h, filename


def main():
    """
    Sanity check that there are no images with the same hash in the datasets.
    """
    filenames = []
    for glob_pattern in GLOB_PATTERNS:
        filenames.extend(glob.glob(glob_pattern, recursive=True))

    os.environ["PYTHONHASHSEED"] = "0"
    hash_filenames = {}

    duplicates = []
    with ThreadPoolExecutor(NUMBER_OF_THREADS) as executor:
        futures = []
        for filename in filenames:
            future = executor.submit(get_image_hash, filename)
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures)):
            hash, filename = future.result()
            if hash in hash_filenames:
                hash_filenames[hash].append(filename)
            else:
                hash_filenames[hash] = [filename]

    duplicates = sorted(filter(
        lambda filenames: len(filenames) > 1, hash_filenames.values()
    ))
    with open(DUPLICATE_JSON_PATH, "w") as f:
        json.dump(duplicates, f, indent=4)
    print(f"Found {len(duplicates)} duplicate images.")
    print(f"See: {DUPLICATE_JSON_PATH}")
    assert len(duplicates) == 0


if __name__ == "__main__":
    main()
