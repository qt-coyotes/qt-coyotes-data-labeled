import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "coco_path",
        type=str,
        help="Path to COCO JSON file.",
    )
    args = parser.parse_args()
    coco_path = Path(args.coco_path)
    with open(coco_path) as f:
        coco = json.load(f)
    images = coco['images']
    unique_w_h = set((i["width"], i["height"]) for i in images)
    print(unique_w_h)


if __name__ == "__main__":
    main()
