from tidecv import TIDE, datasets
import os
from pycocotools.coco import COCO

experiment_path = 'coco_retinanet_baseline_50_1x'

tide = TIDE()
tide.evaluate(datasets.COCO(), datasets.COCOResult(os.path.join(experiment_path, "output/inference/coco_instances_results.json")), mode=TIDE.BOX) # Use TIDE.MASK for masks
tide.summarize()  # Summarize the results as tables in the console
tide.plot(out_dir='tide_plots')       # Show a summary figure. Specify a folder and it'll output a png to that folder.


for i in range(5):
    level_gt_json = f'/media/deepstorage01/datasets_external/coco/annotations/instances_val2017_level_{i+1}.json'
    print(level_gt_json)
    level_gt = COCO(level_gt_json) 
    tide.evaluate(level_gt, datasets.COCOResult(os.path.join(experiment_path, f"output/inference/coco_instances_results_{i}.json")), mode=TIDE.BOX) # Use TIDE.MASK for masks
    tide.summarize()  # Summarize the results as tables in the console`
    tide.plot(out_dir='tide_plots')       # Show a summary figure. Specify a folder and it'll output a png to that folder.
