import numpy as np
from sklearn.utils.extmath import softmax
import torch

from detectron2.data import hierarchical_loss_utils
from detectron2.modeling.meta_arch.retinanet import (
    hierarchical_sigmoid_focal_loss,
    sigmoid_focal_loss_from_probabilities,
)
from detectron2.modeling.roi_heads.fast_rcnn import hierarchical_softmax_cross_entropy
from fvcore.nn import giou_loss, sigmoid_focal_loss_jit


def test_make_hierarchical_matrices():
    content = ["group1/class1", "group1/class2", "group2/class3", "group2/class4", "group3/class5", "group3/class6"]
    results = hierarchical_loss_utils.make_hierarchical_matrices(content)
    assert(len(results) == 1)
    assert(results[0].shape == (6, 3))
    content = ["cat1/group1/class1", "cat1/group1/class2", "cat2/group2/class3", "cat2/group2/class4", "cat2/group3/class5", "cat2/group3/class6"]
    results = hierarchical_loss_utils.make_hierarchical_matrices(content)
    assert (len(results) == 2)
    assert (results[0].shape == (6, 3))
    assert (results[1].shape == (3, 2))


def test_sigmoid_focal_loss_from_probabilities() -> None:
    probs = torch.tensor(
        [
            [0.1, 0.7, 0.1, 0.5, 0.5],
            [0.2, 0.1, 0.8, 0.2, 0.15],
            [0.1, 0.8, 0.1, 0.05, 0.05],
            [0.6, 0.1, 0.7, 0.2, 0.2],
        ],
        dtype=torch.float32,
    )
    label = torch.tensor(
        [[1, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]],
        dtype=torch.float32,
    )

    loss_out = sigmoid_focal_loss_from_probabilities(probs, label, reduction="mean")
    expected_loss = torch.tensor(0.263427)
    np.testing.assert_allclose(loss_out, expected_loss, rtol=1e-05)

    sig_probs = torch.sigmoid(probs)
    loss_out2 = sigmoid_focal_loss_from_probabilities(sig_probs, label, reduction="sum")
    loss_jit = sigmoid_focal_loss_jit(probs, label, reduction="sum")
    np.testing.assert_allclose(loss_jit, loss_out2, rtol=1e-05)

    # hierarchical_matrices = [torch.tensor([[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]),
    #                          torch.tensor([[1, 0], [1, 0], [0, 1]]),
    #                          ]
    # loss_out3 = hierarchical_sigmoid_focal_loss(probs.cuda(), label.cuda(), hierarchical_matrices, reduction="sum")
    # np.testing.assert_allclose(loss_jit, loss_out3, rtol=1e-05)
    # print(loss_jit)


def test_hierarchical_sigmoid_focal_loss() -> None:
    # prediction_logits = torch.tensor(
    #     [
    #         [1.7, 3.1, 4.2, 5.5, 0.5],
    #         [2.2, 4.1, 1.8, -0.2, -0.15],
    #         [0.1, 2.8, 4.1, 5.05, -0.05],
    #         [2.6, -2.1, 1.7, -0.2, 0.2],
    #     ],
    #     dtype=torch.float32,
    # )
    prediction_logits = torch.tensor(
        [
            [0.1, 0.7, 0.1, 0.5, 0.5],
            [0.2, 0.1, 0.8, 0.2, 0.15],
            [0.1, 0.8, 0.1, 0.05, 0.05],
            [0.6, 0.1, 0.7, 0.2, 0.2],
        ],
        dtype=torch.float32,
    )
    label = torch.tensor(
        [[1, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]],
        dtype=torch.float32,
    )
    hierarchical_matrices = [
        np.asarray([[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]),
        np.asarray([[1, 0], [1, 0], [0, 1]]),
    ]

    # hand computed
    expected_loss = torch.tensor(5.873804 * 3)

    loss_out = hierarchical_sigmoid_focal_loss(
        prediction_logits.cuda(), label.cuda(), hierarchical_matrices, reduction="sum"
    )
    np.testing.assert_allclose(loss_out.cpu(), expected_loss, rtol=1e-05)


def test_aggregate_level_targets() -> None:
    targets = torch.tensor([[0, 0, 0, 1, 0], [0, 0, 1, 0, 0]], dtype=torch.float32)
    level_matrix = np.asarray(
        [[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int32
    )
    expected_label = torch.tensor([[0, 1, 0], [0, 1, 0]], dtype=torch.float32)
    aggreagated_label = hierarchical_loss_utils.aggregate_level_targets(
        targets.cuda(), level_matrix
    )
    np.testing.assert_array_almost_equal(
        aggreagated_label.cpu(), expected_label, decimal=5
    )


def test_aggregate_level_probs_with_union() -> None:
    probs = torch.tensor([[0.02, 0.3, 0.2, 0.7, 0.03], [0.05, 0.1, 0.9, 0.2, 0.01]])
    level_matrix = np.asarray([[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]])
    expected_probs = np.asarray(
        [[0.02, 0.832, 0.03], [0.05, 0.928, 0.01]], dtype=np.float32
    )

    ret_probs = hierarchical_loss_utils.aggregate_level_probs_with_union(
        probs.cuda(), level_matrix
    )
    np.testing.assert_array_almost_equal(ret_probs.cpu(), expected_probs, decimal=5)

test_make_hierarchical_matrices()
test_sigmoid_focal_loss_from_probabilities()
test_hierarchical_sigmoid_focal_loss()
test_aggregate_level_targets()
test_aggregate_level_probs_with_union()

def test_convert_scores_by_union():
    unaggregated_scores = np.asarray([0.02, 0.3, 0.2, 0.7, 0.03])
    level_matrix = np.asarray([[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]])
    expected_scores = np.asarray([0.02, 0.832, 0.03], dtype=np.float32)
    ret_scores = hierarchical_loss_utils.convert_scores_by_union(
        unaggregated_scores, level_matrix
    )
    np.testing.assert_array_almost_equal(ret_scores, expected_scores, decimal=5)


def test_hierarchy_mapping():
    level_matrix = np.asarray([[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]])
    expected_mapping = [0, 1, 1, 1, 2]
    out_mapping = hierarchical_loss_utils.hierarchy_mapping(level_matrix)
    np.testing.assert_array_almost_equal(out_mapping, expected_mapping, decimal=5)


def test_():
    json_file = "/home/wanyu/projects/hierachical_loss/hierarchical_loss_detectron/datasets/coco/annotations/instances_val2017.json"
    dataset_name = "coco_2017_val"
    from detectron2.data import MetadataCatalog
    metadata= MetadataCatalog.get(dataset_name)
    _hierarchical_matrices = hierarchical_loss_utils.make_hierarchical_matrices(
        class_hierarchy=MetadataCatalog.get(dataset_name).get("wordnet_hierarchy"),
        add_root_level=True,
        add_background_class=True,
    )
    level_names = metadata.get("wordnet_hierarchy_level_class_names")
    level_names.update({f"{len(level_names) + 1}": ["fg", "bg"]})
    wordnet_hierarchy=MetadataCatalog.get(dataset_name).get("wordnet_hierarchy")
    for level, hierarchical_matrix in enumerate(_hierarchical_matrices):
        # make hierarchy conversion
        if level == 0:
            prev_matrix = hierarchical_matrix
        else:
            prev_matrix = np.matmul(prev_matrix, hierarchical_matrix)
        level_mapping_vector = hierarchical_loss_utils.hierarchy_mapping(prev_matrix)

        hierarchical_loss_utils.make_hierarchical_json(json_file,
                                                       level,
                                                       level_mapping_vector,
                                                       reverse_dict=metadata.thing_dataset_id_to_contiguous_id,
                                                       class_names=level_names[f"{level + 1}"],
                                                       output_dir= "/home/wanyu/home2/hierachical/coco_faster_rcnn/")
        assert True


def get_ref_resulst(hierachical_matrices):
    prediction_logits = np.array([[0.1773, 0.3420, 0.1507, -0.2989],
                                  [-0.5861, -0.2289, -0.0538, 0.0584],
                                  [-0.3863, 0.7643, 0.6831, -0.6814],
                                  [-0.3863, 0.7643, 0.6831, 0.6484],
                                  [0.1197, 0.5120, -0.6425, 0.3661],
                                  [-0.0088, 0.5375, 1.0313, -0.7402]])
    target_class = [1, 1, 3, 2, 0, 2]

    prob_leaf = softmax(prediction_logits)
    loss = -np.log(prob_leaf)
    leaf_level = 0
    for row,column in enumerate(target_class):
        leaf_level += loss[row, column]
    root_loss = leaf_level / 6
    for conversion in hierachical_matrices:
        prob_leaf = np.matmul(prob_leaf, conversion)
        mapping_dict = np.where(conversion > 0)
        target_class = [mapping_dict[1][i] for i in target_class]
        loss = -np.log(prob_leaf)

        for row, column in enumerate(target_class):
            leaf_level += loss[row, column]

    return root_loss, leaf_level / 6


def test_hierarchical_loss():
    # hierarchy =
    # Tree("root", [
    #     Tree("node1",
    #                [Tree("node2", ["node3"]),
    #                Tree("node4", ["node5"]),
    #                Tree("node6", ["node7", "node8"])])])
    # classes = ["node3", "node5", "node7", "node8"]

    content = ["node1/node2/node3", "node1/node4/node5", "node1/node6/node7", "node1/node6/node8"]
    hierarchical_matrices = hierarchical_loss_utils.make_hierarchical_matrices(content)

    inputs = torch.tensor([[0.1773, 0.3420, 0.1507, -0.2989],
                           [-0.5861, -0.2289, -0.0538, 0.0584],
                           [-0.3863, 0.7643, 0.6831,  -0.6814],
                           [-0.3863, 0.7643, 0.6831, 0.6484],
                           [0.1197, 0.5120, -0.6425, 0.3661],
                           [-0.0088, 0.5375, 1.0313,  -0.7402]]).cuda()    # (batch_size, num_classes) 6 x 4
    target = torch.tensor([1, 1, 3, 2, 0, 2]).cuda()   # (batch_size,) 6

    ref_results = get_ref_resulst(hierarchical_matrices)
    from detectron2.utils.events import EventStorage
    with EventStorage() as storage:
        root_level = hierarchical_softmax_cross_entropy(inputs, target, [], [1])
        total_loss = hierarchical_softmax_cross_entropy(inputs, target, hierarchical_matrices, [1, 1, 1])
    np.testing.assert_almost_equal(root_level.cpu().numpy(), np.asarray(ref_results[0], dtype=np.float32), decimal=5)
    np.testing.assert_almost_equal(total_loss.cpu().numpy(), np.asarray(ref_results[1], dtype=np.float32), decimal=5)


if __name__ == "__main__":
    test_()
    test_convert_scores_by_union()
    test_hierarchy_mapping()
    test_make_hierarchical_matrices()
    test_sigmoid_focal_loss_from_probabilities()
    test_hierarchical_sigmoid_focal_loss()
    test_aggregate_level_targets()
    test_aggregate_level_probs_with_union()
