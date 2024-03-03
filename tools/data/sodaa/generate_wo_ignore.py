""" Please enrue this script is in the rawData directory """

import os
import os.path as osp
import json
from tqdm import tqdm


cwd = os.getcwd()
modes = ['train', 'val', 'test']
for mode in modes:
    annWithIgnDir = osp.join(cwd, mode, 'Annotations')
    annDir = osp.join(cwd, mode, 'AnnsWoIgnore')
    os.mkdir(annDir)

    for file in tqdm(sorted(os.listdir(annWithIgnDir)), ncols=120):
        annPth = osp.join(annWithIgnDir, file)
        annFile = json.load(open(annPth, 'r'))
        annotations = annFile['annotations']
        anns = []
        for ann in annotations:
            if int(ann['category_id']) == 9:
                continue
            anns.append(ann)
        annFile['annotations'] = anns
        categories = annFile['categories'][:-1]
        annFile['categories'] = categories
        desPth = osp.join(annDir, file)
        json.dump(annFile, open(desPth, 'w'), indent=4)

