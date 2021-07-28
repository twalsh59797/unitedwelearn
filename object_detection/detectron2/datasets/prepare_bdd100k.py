import cv2
import json
from pathlib import Path


def convert_bdd_coco_format(split) -> None:
    data_folder = Path("/media/deepstorage01/datasets_external/BDD100K")
    json_file = data_folder / "bdd100k_labels_release"  / f"bdd100k_labels_images_{split}.json"
    image_root = Path("/media/deepstorage01/datasets_external/bdd100k/bdd100k/images/100k") / f"{split}"
    with open(json_file) as file:
        results = json.load(file)

    # all_categories = {'traffic sign': 1, 'traffic light': 2, 'car': 3, 'rider': 4, 'motorcycle': 5,
    #                   'pedestrian': 6, 'bus': 7, 'truck': 8, 'bicycle': 9, 'other vehicle': 10,
    #                   'train': 11, 'trailer': 12, 'other person': 13}
    old_categories = {'traffic sign': 9, 'traffic light': 8, 'car': 3, 'rider': 6, 'motor': 4,
                      'person': 7, 'bus': 1, 'truck': 2, 'bike': 5, 'train': 10}
    """
    Group 1: 1-Bus, 2-Truck, 3-Car
    Group 2: 4-Motor, 5-Bike
    Group 3: 6-Rider, 7-Person
    Group 4: 8-Traffic light, 9-Traffic sign
    Group 5: 10-Train
    """
    list_categories = []
    for name, index in  old_categories.items():
        single_cat = {'id': index, 'name': name, 'supercategory': name}
        list_categories.append(single_cat)

    images = []
    annotations = []
    for image_id, img_gt in enumerate(results):
        image_name = img_gt["name"]
        imagepath = image_root / image_name
        img = cv2.imread(str(imagepath))
        height, width, c = img.shape

        img_dict = {
            "license": 0,
            "file_name": image_name,
            "coco_url": "",
            "height": height,
            "width": width,
            "data_captured": "",
            "id": image_id,
        }
        images.append(img_dict)

        detection_gt = img_gt["labels"] if img_gt["labels"] is not None else []
        for bbox_i in detection_gt:
            label_id = old_categories.get(bbox_i["category"], -1)
            if label_id > 0:
                coord = bbox_i["box2d"]
                # coco format: [x1, y1, width, height] in pixel
                coco_box = [coord["x1"], coord["y1"], coord["x2"] - coord["x1"], coord["y2"] - coord["y1"]]
                box_area = coco_box[2] * coco_box[3]
                # ignore the box which is not included in the categories defined
                annotations_dict = {
                    "segmentation": {"counts": None},  # to support usage of tide analysis
                    "area": box_area,
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": coco_box,
                    "category_id": label_id,
                    "id": bbox_i["id"],
                }
                annotations.append(annotations_dict)


    json_file = data_folder / "bdd100k_labels_detection20"  / f"det_v1_{split}_detectron2_format.json"
    with json_file.open("w") as file:
        instances = {
            "annotations": annotations,
            "images": images,
            "categories": list_categories,
        }
        json.dump(instances, file, indent=2)


if __name__ == "__main__":
    convert_bdd_coco_format(split="val")
    convert_bdd_coco_format(split="train")