import os
import json
import zipfile
import requests
from tqdm import tqdm
from pathlib import Path

DATA_DIR = Path("data")
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_VAL_IMAGES_URL  = "http://images.cocodataset.org/zips/val2017.zip"


def download_file(url: str, dest: Path):
    if dest.exists():
        print(f"[skip] {dest.name} already exists")
        return
    print(f"Downloading {dest.name} ...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def download_annotations():
    
    zip_path = DATA_DIR / "annotations_trainval2017.zip"
    download_file(COCO_ANNOTATIONS_URL, zip_path)
    ann_dir = DATA_DIR / "annotations"
    if not ann_dir.exists():
        print("Extracting annotations...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(DATA_DIR)
    print(f"[ok] Annotations at {ann_dir}")


def build_mini_subset(num_images: int = 500, out_json: str = "data/coco_mini500.json"):
    src_json = DATA_DIR / "annotations" / "instances_val2017.json"
    assert src_json.exists(), "Run download_annotations() first"

    with open(src_json) as f:
        coco = json.load(f)

    selected_ids = {img["id"] for img in coco["images"][:num_images]}
    mini = {
        "info":        coco.get("info", {}),
        "licenses":    coco.get("licenses", []),
        "images":      [img for img in coco["images"] if img["id"] in selected_ids],
        "annotations": [ann for ann in coco["annotations"] if ann["image_id"] in selected_ids],
        "categories":  coco["categories"],
    }

    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(mini, f)
    print(f"[ok] Mini subset: {len(mini['images'])} images, "
          f"{len(mini['annotations'])} annotations → {out}")
    return str(out)


def download_val_images():
    zip_path = DATA_DIR / "val2017.zip"
    download_file(COCO_VAL_IMAGES_URL, zip_path)
    img_dir = DATA_DIR / "val2017"
    if not img_dir.exists():
        print("Extracting images...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(DATA_DIR)
    print(f"[ok] Images at {img_dir} ({len(list(img_dir.iterdir()))} files)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Also download all 5000 val2017 images (~1 GB)")
    parser.add_argument("--subset-size", type=int, default=500,
                        help="Number of images in the mini subset (default: 500)")
    args = parser.parse_args()

    download_annotations()
    build_mini_subset(num_images=args.subset_size)

    if args.full:
        download_val_images()
    else:
        print("\nTo download val images: python download_data.py --full \n Or download only the images you need using the image IDs in data/coco_mini500.json")
