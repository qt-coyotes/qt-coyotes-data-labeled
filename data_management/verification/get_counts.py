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
    parser.add_argument(
        "--categories_mange",
        type=int,
        nargs='+',
        help="Category IDs for mange.",
        required=True
    )
    parser.add_argument(
        "--categories_nomange",
        type=int,
        nargs='+',
        help="Category IDs for nomange.",
        required=True
    )
    args = parser.parse_args()
    coco_path = Path(args.coco_path)
    with open(coco_path) as f:
        coco = json.load(f)

    categories_mange = set(args.categories_mange)
    categories_nomange = set(args.categories_nomange)

    n_mange = 0
    n_nomange = 0
    for annotation in coco['annotations']:
        if annotation["category_id"] in categories_mange:
            n_mange += 1
        elif annotation["category_id"] in categories_nomange:
            n_nomange += 1

    n_total = n_mange + n_nomange
    print(f"[number with mange / total number]: [{n_mange}, {n_total}]")


if __name__ == "__main__":
    main()
