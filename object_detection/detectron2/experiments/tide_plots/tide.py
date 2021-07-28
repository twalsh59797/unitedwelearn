from tidecv import TIDE, datasets
import os

experiment_path = '../coco_retinanet_baseline_50_1x'

tide = TIDE()
# tide.evaluate(datasets.COCO(), datasets.COCOResult(os.path.join(experiment_path, "output/inference/coco_instances_results.json")), mode=TIDE.BOX) # Use TIDE.MASK for masks
# tide.summarize()  # Summarize the results as tables in the console
# tide.plot(out_dir='tide_plots_base')       # Show a summary figure. Specify a folder and it'll output a png to that folder.


# for i in range(9):
#     level_gt_json = f'/media/deepstorage01/datasets_external/coco/annotations/instances_val2017_level_{i}.json'
#    print(level_gt_json)
#     level_gt = datasets.COCO(path=level_gt_json) 
#     tide.evaluate(level_gt, datasets.COCOResult(os.path.join(experiment_path, f"output/inference/coco_instances_results_{i}.json")), mode=TIDE.BOX) # Use TIDE.MASK for masks
#     tide.summarize()  # Summarize the results as tables in the console`
#     tide.plot(out_dir='tide_plots_base')       # Show a summary figure. Specify a folder and it'll output a png to that folder.
tide.evaluate(datasets.COCO(), datasets.COCOResult(os.path.join(experiment_path, "output/inference/coco_instances_results.json")), mode=TIDE.BOX)
tide.summarize()

for i in range(9):
    tide=TIDE()
    level_gt_json = f'/media/deepstorage01/datasets_external/coco/annotations/instances_val2017_level_{i}.json'
    level_gt = datasets.COCO(path=level_gt_json)
    tide.evaluate(level_gt, datasets.COCOResult(os.path.join(experiment_path, f"output/inference/coco_instances_results_{i}.json")), mode=TIDE.BOX) # Use TIDE.MASK for masks
    tide.summarize()

experiment_path = '../coco_retinanet_hier_50_1x_w080101'

tide=TIDE()
tide.evaluate(datasets.COCO(), datasets.COCOResult(os.path.join(experiment_path, "output/inference/coco_instances_results.json")), mode=TIDE.BOX) # Use TIDE.MASK for masks
tide.summarize()

for i in range(9):
    tide=TIDE()
    level_gt_json = f'/media/deepstorage01/datasets_external/coco/annotations/instances_val2017_level_{i}.json'
    level_gt = datasets.COCO(path=level_gt_json)
    tide.evaluate(level_gt, datasets.COCOResult(os.path.join(experiment_path, f"output/inference/coco_instances_results_{i}.json")), mode=TIDE.BOX) # Use TIDE.MASK for masks
    tide.summarize()


# tide = TIDE()
# tide.evaluate(datasets.COCO(), datasets.COCOResult(os.path.join(experiment_path, "output/inference/coco_instances_results.json")), mode=TIDE.BOX) # Use TIDE.MASK for masks
# tide.summarize()  # Summarize the results as tables in the console
# tide.plot(out_dir='tide_plots_hier')       # Show a summary figure. Specify a folder and it'll output a png to that folder.


# for i in range(5, 9):
#     level_gt_json = f'/media/deepstorage01/datasets_external/coco/annotations/instances_val2017_level_{i}.json'
#     print(level_gt_json)
#     level_gt = datasets.COCO(path=level_gt_json)
#   tide.evaluate(level_gt, datasets.COCOResult(os.path.join(experiment_path, f"output/inference/coco_instances_results_{i}.json")), mode=TIDE.BOX) # Use TIDE.MASK for masks
#    tide.summarize()  # Summarize the results as tables in the console`
#    tide.plot(out_dir='tide_plots_hier')       # Show a summary figure. Specify a folder and it'll output a png to that folder.

