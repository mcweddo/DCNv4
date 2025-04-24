# fast_polygon_validator.py  --------------------------------------------
from pycocotools.coco import COCO
import math, tqdm

ANNOT = "data/coco/annotations/instances_train.json"   # adjust this path

def poly_is_bad(seg):
    """
    Return None if polygon list is OK, else a short reason string.
    """
    if not seg:                     # empty list
        return "empty"
    for poly in seg:
        if len(poly) < 6 or (len(poly) & 1):
            return "len<6-or-odd"   # need ≥3 points and even length
        if any(math.isnan(x) for x in poly):
            return "nan"
    return None

def main():
    coco = COCO(ANNOT)              # prints "...index loaded."
    bad = []

    for ann in tqdm.tqdm(coco.dataset["annotations"]):
        reason = poly_is_bad(ann["segmentation"])
        if reason:
            img_id   = ann["image_id"]
            file_n   = coco.imgs[img_id]["file_name"]
            bad.append((ann["id"], file_n, reason))

    print(f"\nFound {len(bad)} broken annotations")
    for ann_id, file_n, reason in bad[:30]:
        print(f"{ann_id:>8}  {file_n:<40}  -> {reason}")

if __name__ == "__main__":
    main()
