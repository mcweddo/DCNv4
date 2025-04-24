from pycocotools.coco import COCO
import numpy as np, json, tqdm

coco = COCO('detection/data/coco/annotations/instances_train.json')
bad_anns = []
for ann_id in tqdm.tqdm(coco.getAnnIds()):
    ann = coco.loadAnns([ann_id])[0]
    seg  = ann['segmentation']
    if not seg:                      # empty list
        bad_anns.append((ann_id, 'empty'))
        continue
    for poly in seg:
        if len(poly) % 2 or len(poly) < 6:
            bad_anns.append((ann_id, 'invalid len'))
            break
        if np.any(np.isnan(poly)):
            bad_anns.append((ann_id, 'NaN'))
            break

print('bad ann ids:', bad_anns[:10])

