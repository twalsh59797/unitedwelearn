from nltk import Tree
import numpy as np
import torch


def make_hierarchical_matrices(classes, hierarchy):
    assert set(classes) == set(hierarchy.leaves())
    level_dicts = []
    level_hierarchy = hierarchy

    for level in range(hierarchy.height() - 2):
        level_dict = {}
        next_level_tree = Tree('root', [])
        for parent in level_hierarchy:
            if not isinstance(parent, str):
                for child in parent:
                    if not isinstance(child, str):
                        level_dict[child.label()] = parent.label()
                    else:
                        level_dict[child] = parent.label()
                    next_level_tree.append(child)
        level_dicts.append(level_dict)
        level_hierarchy = next_level_tree

    hierarchical_matrices = []
    curr_level_classes = classes
    for level in range(len(level_dicts) - 1, -1, -1):
        curr_dict = level_dicts[level]
        level_matrix = np.zeros((len(curr_dict.keys()), len(set(curr_dict.values()))))
        parent_to_idx = {k: v for (k, v) in
                         zip(sorted(set(curr_dict.values())), range(len(set(curr_dict.values()))))}
        for cl_idx, cl in enumerate(curr_level_classes):
            parent_idx = parent_to_idx[curr_dict[cl]]
            level_matrix[cl_idx, parent_idx] = 1
        curr_level_classes = sorted(set(curr_dict.values()))
        hierarchical_matrices.append(level_matrix)

    return hierarchical_matrices

def test_hierarchical_matrices():
    hierarchy = Tree("root", [Tree("node1", [Tree("node2", ["node3"]), Tree("node4", ["node5"]), Tree("node6", ["node7", "node8"])])])
    classes = ["node3", "node5", "node7", "node8"]

    hierarchical_matrices = make_hierarchical_matrices(classes, hierarchy)

    h0 = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]])
    h1 = np.asarray([[1], [1], [1]])
    assert hierarchical_matrices[0].shape == (4, 3)
    assert hierarchical_matrices[1].shape == (3, 1)

    np.testing.assert_array_almost_equal(h0, hierarchical_matrices[0])
    np.testing.assert_array_almost_equal(h1, hierarchical_matrices[1])


def hierarchical_loss(inputs, target, hierarchical_matrices):
    total_loss = 0
    for level_i in range(len(hierarchical_matrices) + 1):
        if level_i == 0:
            level_inputs, level_target = inputs, target
        else:
            level_transform = torch.from_numpy(hierarchical_matrices[level_i - 1]).float()
            level_transform = level_transform.cuda()
            level_inputs = torch.matmul(prev_level_inputs, level_transform)
            mapping = np.argmax(hierarchical_matrices[level_i - 1], axis=1)
            mapping = torch.from_numpy(mapping)
            mapping = mapping.cuda()
            level_target = mapping[prev_level_target]

        loss_level = torch.nn.functional.cross_entropy(level_inputs, level_target)
        total_loss += loss_level

        prev_level_inputs, prev_level_target = level_inputs, level_target

    return total_loss


def test_hierarchical_loss():
    hierarchy = Tree("root", [
        Tree("node1", [Tree("node2", ["node3"]), Tree("node4", ["node5"]), Tree("node6", ["node7", "node8"])])])
    classes = ["node3", "node5", "node7", "node8"]

    hierarchical_matrices = make_hierarchical_matrices(classes, hierarchy)

    inputs = torch.tensor([[0.1773, 0.3420, 0.1507, -0.2989],
                           [-0.5861, -0.2289, -0.0538, 0.0584],
                           [-0.3863, 0.7643, 0.6831,  -0.6814],
                           [-0.3863, 0.7643, 0.6831, 0.6484],
                           [0.1197, 0.5120, -0.6425, 0.3661],
                           [-0.0088, 0.5375, 1.0313,  -0.7402]]).cuda()     # (batch_size, num_classes) 6 x 4
    target = torch.tensor([1, 1, 3, 2, 0, 2]).cuda()   # (batch_size,) 6


    total_loss = hierarchical_loss(inputs, target, hierarchical_matrices)
    expected_loss = np.asarray(2.419285, dtype=np.float32)
    np.testing.assert_almost_equal(total_loss.cpu().numpy(), expected_loss, decimal=5)

    #############################################
    inputs = torch.tensor(
        [
            [0.1, 0.7, 0.1, 0.05, 0.05],
            [0.01, 0.1, 0.79, 0.05, 0.05],
            [0.01, 0.1, 0.79, 0.05, 0.05],
            [0.01, 0.1, 0.79, 0.05, 0.05],
        ]
    ).cuda()  # batch x num classes (4, 5)
    target = torch.tensor([0, 3, 1, 2]).cuda()
    hierarchical_matrices = (np.asarray([[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]),
                             np.asarray([[1, 0], [1, 0], [0, 1]]))
    total_loss = hierarchical_loss(inputs, target, hierarchical_matrices)
    # hand computed
    expected_loss = torch.tensor(3.18081)
    np.testing.assert_almost_equal(total_loss.cpu().numpy(), expected_loss, decimal=4)


test_hierarchical_matrices()
test_hierarchical_loss()
