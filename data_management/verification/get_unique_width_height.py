import json
from pathlib import Path


COCO_PATH = Path("data/processed/mange_Toronto/mange_Toronto.json")
COCO_PATH = Path("data/processed/mange_images/mange_images.json")


def main():
    with open(COCO_PATH) as f:
        coco = json.load(f)
    images = coco['images']
    unique_w_h = set((i["width"], i["height"]) for i in images)
    print(unique_w_h)


if __name__ == "__main__":
    main()
