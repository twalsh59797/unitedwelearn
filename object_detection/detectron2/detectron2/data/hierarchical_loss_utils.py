import copy
import itertools
import json
import numpy as np
import os
import torch


def make_hierarchical_matrices(class_hierarchy, add_background_class=False, add_root_level: bool = False):
    """Makes binary matrixes of mapping for the hierarchical
    loss computation."""
    if class_hierarchy is None:
        return None, None

    label_names = list(copy.deepcopy(class_hierarchy))
    num_levels = len(label_names[0].split("/"))
    if add_background_class:
        label_names.append("background")

    hierachical_matrices = []
    children_dict = make_value_dict(label_names)
    for level_i in range(0, num_levels-1):
        parent_dict = _hierarchy_parent_from_children(children_dict, get_label_parents)
        children_to_parent_vector = hierarchy_dict_to_vector(
            children_dict, parent_dict, get_label_parents
        )
        children_to_parent_transform = hierarchy_one_hot(
            children_to_parent_vector,
            len(children_dict.keys()),
            len(parent_dict.keys()),
        )
        hierachical_matrices.append(children_to_parent_transform)
        children_dict = parent_dict
    if add_root_level:
        if add_background_class:
            map_to_fg_bg = np.zeros(shape=(len(children_dict)), dtype = np.int32)
            map_to_fg_bg[-1] = 1
            to_root_transform = hierarchy_one_hot(map_to_fg_bg, len(children_dict), 2)
        else:
            to_root_transform = np.ones(shape = (len(children_dict), 1), dtype = np.int32)
        hierachical_matrices.append(to_root_transform)
    return hierachical_matrices


def make_value_dict(content_list):
    level_dict = {}
    for label_id, label in enumerate(content_list):
        level_dict[label] = label_id
    return level_dict


def _hierarchy_parent_from_children(child_dict, parent_function):
    """
    Makes parent level dictionary from a child-level one and the function that strips
    the extended child label to the respective parent one..
    Sorts parent_dict keys based on the child dict,
    for example, if the children classes are
    {"b/c/d": 0, "a/e/f": 1, "b/e": 2, "b/c/r": 3},
    the parent dict becomes {"b/c": 0, "a/e": 1, "b/e": 2}.
    """
    level_dict = {}
    sorted_key_val = sorted(child_dict.items(), key=lambda item: item[1])
    node_id = 0
    for node, _ in sorted_key_val:
        parent = parent_function(node)
        if parent not in level_dict.keys():
            level_dict[parent] = node_id
            node_id += 1
    return level_dict


def hierarchy_one_hot(level_mapping_vector, len_level, len_next_level):
    level_transform = np.zeros(shape=(len_level, len_next_level), dtype=np.int32)
    level_transform[np.arange(len_level), level_mapping_vector] = 1
    return level_transform


def get_label_parents(label_str):
    label_split = label_str.split("/")
    return "/".join(label_split[0:len(label_split) - 1])


def get_label_category(label_str: str) -> str:
    label_split = label_str.split("/")
    return label_split[0]


def hierarchy_dict_to_vector(level_dict, next_level_dict, parent_function):
    level_mapping_vector = np.zeros(len(level_dict.keys()), dtype=np.int32)
    for value in level_dict.keys():
        level_mapping_vector[level_dict[value]] = next_level_dict[
            parent_function(value)
        ]
    return level_mapping_vector


def aggregate_level_targets(targets, level_matrix):
    """Translates child-level labels to parent-level labels with the hierarchical transition matrix
    :param targets: batch_size x num_children_classes tensor of labels
    :param level_matrix: num_children_classes x num_parent_classes binary transition matrix"""
    t_level_matrix = torch.from_numpy(level_matrix).float()
    return torch.matmul(targets, t_level_matrix.cuda())


def aggregate_level_probs_with_union(probs, level_matrix):
    """Aggregates predicted probabilities to hierarchical parent probabilities with probability unions
    and inclusion-exclusion principle
    :param probs: batch_size x num_children_classes tensor of probabilities
    :param level_matrix: num_children_classes x num_parent_classes binary transition matrix"""
    num_children_per_class = np.sum(level_matrix, axis=0)
    num_parent_classes = level_matrix.shape[1]
    batch_size = probs.size()[0]

    # Step 1: sum all probabilities
    t_level_matrix = torch.from_numpy(level_matrix).float()
    res_probs = torch.matmul(
        probs, t_level_matrix.cuda()
    )  # batch_size x num_children_classes
    for class_idx in range(num_parent_classes):
        num_children_classes = num_children_per_class[class_idx]
        # From step 2, inclusion-exclusion principle: subtract pairs, add triples, subtract quadruples etc
        if num_children_classes > 1:
            class_update = torch.zeros((batch_size,)).cuda()
            class_probability_indices = np.where(level_matrix[:, class_idx] == 1)[0]
            class_probs = torch.stack(
                [probs[:, idx] for idx in class_probability_indices], dim=-1
            )

            for child_class_idx in range(1, num_children_classes):
                coef = 1 if child_class_idx % 2 == 0 else -1
                for level_perm in itertools.combinations(
                    range(num_children_classes), child_class_idx + 1
                ):
                    level_pair_probs = torch.stack(
                        [class_probs[:, idx] for idx in level_perm], dim=-1
                    )
                    level_pair_probs = torch.prod(level_pair_probs, axis=-1)
                    class_update += level_pair_probs * coef
            res_probs[:, class_idx] += class_update

    return res_probs


def convert_scores_by_union(unaggregated_scores, hierarchical_matrix):
    """
    Args:
        unaggregated_scores: 1 x K scores
        hierarchical_matrix: K x N matrix
    Returns: 1 x N Converted scores
    """
    num_classes = hierarchical_matrix.shape[1]  # N
    num_children_per_class = np.sum(hierarchical_matrix, axis=0)

    ################################
    # Step 1: sum all probabilities
    converted_scores = np.matmul(unaggregated_scores, hierarchical_matrix)
    for class_idx in range(num_classes):
        num_children_classes = num_children_per_class[class_idx]
        # From step 2, inclusion-exclusion principle: subtract pairs, add triples, subtract quadruples etc
        if num_children_classes > 1:
            class_update = 0
            class_probability_indices = np.where(
                hierarchical_matrix[:, class_idx] == 1
            )[0]
            class_probs = np.array(unaggregated_scores)[class_probability_indices]
            for child_class_idx in range(1, num_children_classes):
                coef = 1 if child_class_idx % 2 == 0 else -1
                for level_perm in itertools.combinations(
                    range(num_children_classes), child_class_idx + 1
                ):  # generator
                    level_pair_probs = class_probs[np.array(level_perm)]
                    level_pair_prob = np.prod(level_pair_probs)
                    class_update += level_pair_prob * coef
            converted_scores[class_idx] += class_update
    ################################
    return converted_scores


def convert_gt_hierarchical(coco_gt, children_to_level_mapping, reverse_dict):
    for key in coco_gt.anns.keys():
        coco_gt.anns[key]["category_id"] = children_to_level_mapping[
            reverse_dict[coco_gt.anns[key]["category_id"]]
        ]
    return coco_gt


def convert_results_hierarchical(coco_results, hierarchical_matrix, reverse_dict, convert_by_union: bool = True):
    mapping_vector = np.argmax(hierarchical_matrix, axis=1)
    for result in coco_results:
        if reverse_dict:
            result["category_id"] = int(mapping_vector[reverse_dict[result["category_id"]]])
        else:
            result["category_id"] = int(mapping_vector[result["category_id"]])
        if convert_by_union:
            level_scores = convert_scores_by_union(
                result["unaggregated_score"], hierarchical_matrix
            )
        else:
            level_scores = np.matmul(result["unaggregated_score"], hierarchical_matrix)
        result["unaggregated_score"] = list(level_scores)
        result["score"] = result["unaggregated_score"][result["category_id"]]
    return coco_results


def hierarchy_mapping(children_to_level_hierarchical_matrix):
    mapping_vector = np.argmax(children_to_level_hierarchical_matrix, axis=1)
    return mapping_vector


def make_dataset_categories(wordnet_hierarchy, level):
    label_names = list(copy.deepcopy(wordnet_hierarchy))
    level_dict = make_value_dict(label_names)
    for i in range(level):
        level_dict = _hierarchy_parent_from_children(level_dict, get_label_parents)

    dataset_categories = []
    for key in level_dict.keys():
        name = key
        id = level_dict[key]
        supercategory = ""
        dataset_categories.append(
            {"supercategory": supercategory, "id": id, "name": name}
        )
    return dataset_categories


def make_hierarchical_json(json_file, level, children_to_level_mapping, reverse_dict, class_names, regenerate_gt_per_level: bool = False):
    level_filename = os.path.join(os.path.dirname(json_file),
                                  f"{os.path.basename(json_file).split('.')[0]}_level_{level}.json")
    if os.path.exists(level_filename) and not regenerate_gt_per_level:
        return level_filename
    with open(json_file, "r") as jsonFile:
        data = json.load(jsonFile)

    data["categories"] = [{"supercategory": "", "id": idx, "name": name} for idx, name in enumerate(class_names)]

    for ann in data["annotations"]:
        ann["category_id"] = int(children_to_level_mapping[reverse_dict[ann["category_id"]]])
    with open(level_filename, "w") as jsonFile:
        json.dump(data, jsonFile)
    return level_filename
