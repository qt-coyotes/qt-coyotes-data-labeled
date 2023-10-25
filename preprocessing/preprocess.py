import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2 as cv
import numpy as np
from torchvision import transforms as T
from torchvision.io import read_image
from torchvision.transforms import functional as F
from tqdm import tqdm
from transforms import SquarePad


def preprocess(
    annotation: dict,
    raw_image_path: Path,
    processed_image_path: Path,
    crop_size: int,
):
    try:
        image = read_image(str(raw_image_path))
    except RuntimeError:
        return
    bbox = annotation["bbox"]
    x, y, w, h = bbox
    y = int(np.floor(y))
    x = int(np.floor(x))
    h = int(np.ceil(h))
    w = int(np.ceil(w))
    transform = T.Compose(
        [
            SquarePad(),
            T.Resize(
                (crop_size, crop_size),
                antialias=True,
            )
        ]
    )
    image = transform(image)
    processed_image_path.parent.mkdir(parents=True, exist_ok=True)
    image = image.permute(1, 2, 0)
    image = image.numpy()
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    cv.imwrite(str(processed_image_path), image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("coco_path", type=str)
    parser.add_argument("raw_data_path", type=str)
    parser.add_argument("processed_data_path", type=str)
    parser.add_argument("--processes", default=8, required=False, type=str)
    parser.add_argument("--crop_size", default=224, required=False, type=str)
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
    with ProcessPoolExecutor(args.processes) as executor:
        futures = []
        for annotation in annotations:
            image_id = annotation["image_id"]
            image = image_id_to_image[image_id]
            file_name = image["file_name"]
            raw_image_path = raw_data_path / "megadetected" / file_name
            processed_image_path = processed_data_path / file_name
            future = executor.submit(
                preprocess,
                annotation,
                raw_image_path,
                processed_image_path,
                args.crop_size,
            )
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()


if __name__ == "__main__":
    main()
