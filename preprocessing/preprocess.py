import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import cv2
from pathlib import Path


def crop(image: np.ndarray, x: float, y: float, w: float, h: float):
    x = int(np.floor(x))
    y = int(np.floor(y))
    w = int(np.ceil(w))
    h = int(np.ceil(h))
    return image[y : y + h, x : x + w]


def preprocess(
    annotation: dict, raw_image_path: Path, processed_image_path: Path
):
    image = cv2.imread(str(raw_image_path))
    bbox = annotation["bbox"]
    image = crop(image, *bbox)
    cv2.imwrite(str(processed_image_path), image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("coco_path", type=str)
    parser.add_argument("raw_data_path", type=str)
    parser.add_argument("processed_data_path", type=str)
    parser.add_argument("--threads", default=64, required=False, type=str)
    args = parser.parse_args()
    coco_path = Path(args.coco_path)
    raw_data_path = Path(args.raw_data_path)
    processed_data_path = Path(args.processed_data_path)

    processed_data_path.mkdir(parents=True, exist_ok=True)

    with open(coco_path, "r") as f:
        coco = json.load(f)

    image_id_to_image = {image["id"]: image for image in coco["images"]}
    annotations = coco["annotations"]

    # https://stackoverflow.com/questions/51601756/use-tqdm-with-concurrent-futures
    with ThreadPoolExecutor(args.threads) as executor:
        futures = []
        for annotation in annotations:
            image_id = annotation["image_id"]
            image = image_id_to_image[image_id]
            file_name = image["file_name"]
            raw_image_path = raw_data_path / file_name
            processed_image_path = processed_data_path / file_name
            future = executor.submit(
                preprocess, annotation, raw_image_path, processed_image_path
            )
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()


if __name__ == '__main__':
    main()
