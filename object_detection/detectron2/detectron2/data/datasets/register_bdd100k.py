# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import copy
from pathlib import Path
from typing import Dict, List, Tuple

from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances, load_coco_json



def register_bdd100k_instances(json_file: str,  image_root: str, dataset_name: str) -> None:
    """
    Register surface_signs in json annotation format for detection.
    """
    register_coco_instances(dataset_name, {}, str(json_file), image_root)
    split = dataset_name.split("_"[-1])
    class_mapping = {'traffic sign': 9, 'traffic light': 8, 'car': 3, 'rider': 6, 'motor': 4,
                  'person': 7, 'bus': 1, 'truck': 2, 'bike': 5, 'train': 10}
    class_name= [k for k, v in sorted(class_mapping.items(), key=lambda item: item[1])]
    id_mapping = {v: v - 1 for k, v in class_mapping.items()}
    wordnet_hierarchy_level_class_names = {"0" : class_name}
    wordnet_hierarchy, wordnet_hierarchy_level_class_names = _make_bdd_hierarchy(class_mapping, wordnet_hierarchy_level_class_names)

    MetadataCatalog.get(dataset_name).set(
        thing_classes=class_name, dirname="", split=split, thing_dataset_id_to_contiguous_id=id_mapping,
        wordnet_hierarchy=wordnet_hierarchy, wordnet_hierarchy_level_class_names=wordnet_hierarchy_level_class_names

    )


def _make_bdd_hierarchy(categories, wordnet_hierarchy_level_class_names):
    """
    Group 1: 1-Bus, 2-Truck, 3-Car
    Group 2: 4-Motor, 5-Bike
    Group 3: 6-Rider, 7-Person
    Group 4: 8-Traffic light, 9-Traffic sign
    Group 5: 10-Train
    """
    parent_mapping = {
                      1: 1, 2:1, 3: 1,
                      4:2, 5:2,
                      6:3, 7:3,
                      8:4, 9: 4,
                      10: 5,
                      }
    content = []
    for value, value_id in categories.items():
        subcategory = f"group_{parent_mapping[value_id]}"
        content.append(f"{subcategory}/{value}")
    parent_ids = list(set(parent_mapping.values()))
    parent_ids.sort()
    wordnet_hierarchy_level_class_names.update({"1": [f"group_{id}" for id in parent_ids]})
    return content, wordnet_hierarchy_level_class_names


def load_bdd100k_json(json_file, image_root, dataset_name=None):
    return load_coco_json(json_file, image_root, dataset_name)


def visualize():
    """
    Test the bdd100k json dataset loader.

    Usage:
        python -m detectron2.data.datasets.dota \
            path/to/json path/to/image_root dataset_name vis_limit
    """
    import sys
    import numpy as np
    from detectron2.utils.logger import setup_logger
    from PIL import Image
    import detectron2.data.datasets  # noqa # add pre-defined metadata
    from detectron2.utils.visualizer import Visualizer
    import json

    logger = setup_logger(name=__name__)
    json_file = "/media/deepstorage01/datasets_external/bdd100k/bdd100k_labels_release/det_v1_val_detectron2_format.json"
    image_root = "/media/deepstorage01/datasets_external/bdd100k/bdd100k/images/100k/val"
    dataset_name = "bdd100k_val"
    meta = MetadataCatalog.get(dataset_name)

    dicts = load_bdd100k_json(json_file, image_root, dataset_name)
    logger.info("Done loading {} samples.".format(len(dicts)))

    dirname = Path("/home/wanyu/tmp/dota-data-vis")
    dirname.mkdir(parents=True, exist_ok=True)
    for d in dicts[0: 5]:
        img = np.array(Image.open(d["file_name"]))
        visualizer = Visualizer(img, metadata=meta)
        vis = visualizer.draw_dataset_dict(d)
        fpath = dirname / Path(d["file_name"]).name
        vis.save(str(fpath))



if __name__ == "__main__":
    visualize()
