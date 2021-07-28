from tidecv import TIDE, datasets
import json
import sys


def main():
    model_name = "coco_faster_rcnn_2lr"
    pred_folder = f"/home/wanyu/home2/hierachical/{model_name}/inference/"

    if "coco" in model_name:
        num_levels = 11
        json_files = ["instances_val2017.json"] + [f"instances_val2017_level_{l}.json" for l in range(num_levels -1)]
        anno_folder = "/media/deepstorage01/datasets_external/coco/annotations/"
    else:
        num_levels = 3
        json_files = ["det_v1_val_detectron2_format.json"] + [f"det_v1_val_detectron2_format_{l}.json" for l in range(num_levels -1)]
        anno_folder = "/media/deepstorage01/datasets_external/bdd100k/bdd100k_labels_release/"

    predictions = ["coco_instances_results.json"] + [f"coco_instances_results_{l}.json" for l in
                                                     range(num_levels - 1)]

    sys.stdout = open(pred_folder + "log.txt", 'w')

    for level, gt_file_name in enumerate(json_files):
        if level < 10:
            continue

        gt_file = anno_folder + gt_file_name
        with open(gt_file) as f:
            gts = json.load(f)
        # for box_i in gts["annotations"]:
        #     box_i['segmentation'] = {"counts": None}
        # with open(gt_file, "w") as f:
        #     json.dump(gts, f)
        pred_file = pred_folder + predictions[level]
        tide = TIDE()
        tide.evaluate(datasets.COCO(gt_file), datasets.COCOResult(pred_file),
                      mode=TIDE.BOX)  # Use TIDE.MASK for masks
        tide.summarize()  # Summarize the results as tables in the console
        tide.plot(pred_folder)  # Show a summary figure. Specify a folder and it'll output a png to that folder.


if __name__ == "__main__":
    main()