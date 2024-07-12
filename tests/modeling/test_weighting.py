import numpy as np
from numpy.testing import assert_array_almost_equal

from generalize.model.utils.weighting import calculate_sample_weight, calculate_weights


def test_weight_calculation():
    # Balanced y
    y = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])

    # Test weight calculations
    unique_ys, y_weights = calculate_weights(v=y)
    assert_array_almost_equal(unique_ys, [0, 1])
    assert_array_almost_equal(y_weights, [1, 1])

    # Unbalanced y
    y = np.asarray([0, 0, 1, 1, 1, 1, 1, 1])

    # Test weight calculations
    unique_ys, y_weights = calculate_weights(v=y)
    assert_array_almost_equal(unique_ys, [0, 1])
    assert_array_almost_equal(y_weights, [1.5, 0.5])

    # Unbalanced groups
    groups = [0, 0, 0, 0, 2, 2, 3, 3]

    # Calculate expected weights
    exp_inv_freqs = np.array([1 / 4, 1 / 2, 1 / 2])
    exp_weights = exp_inv_freqs / exp_inv_freqs.sum() * 3  # 3 groups

    # Test weight calculations
    unique_groups, group_weights = calculate_weights(v=groups)
    assert_array_almost_equal(unique_groups, [0, 2, 3])
    assert_array_almost_equal(group_weights, exp_weights)

    unique_groups, group_weights = calculate_weights(v=groups, normalize=False)
    assert_array_almost_equal(unique_groups, [0, 2, 3])
    assert_array_almost_equal(group_weights, exp_inv_freqs)


def test_weighting_balanced():
    # Balanced
    y = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    groups = [0, 0, 1, 1, 2, 2, 3, 3]

    weights, new_groups = calculate_sample_weight(
        y=y,
        groups=groups,
        weight_loss_by_class=True,
        weight_loss_by_groups=True,
        weight_per_split=False,
        weight_splits_equally=False,
    )

    # Both groups and labels are balanced
    # So all elements should have the weight 1
    assert_array_almost_equal(weights, [1, 1, 1, 1, 1, 1, 1, 1])

    _check_same_weights(y, weights)
    assert_array_almost_equal(
        new_groups, groups  # No splits in the original groups so no difference
    )


def test_weighting_unbalanced_y():
    # Unbalanced y
    y = np.asarray([0, 0, 1, 1, 1, 1, 1, 1])
    groups = [0, 0, 1, 1, 2, 2, 3, 3]
    weights, new_groups = calculate_sample_weight(
        y=y,
        groups=groups,
        weight_loss_by_class=True,
        weight_loss_by_groups=True,
        weight_per_split=False,
        weight_splits_equally=False,
    )

    assert_array_almost_equal(
        weights,
        [
            2.0,
            2.0,
            0.666666667,
            0.666666667,
            0.666666667,
            0.666666667,
            0.666666667,
            0.666666667,
        ],
    )
    # Test each label has same sum of weights
    _check_same_weights(y, weights)
    assert_array_almost_equal(
        new_groups, groups  # No splits in the original groups so no difference
    )


def test_weighting_unbalanced_groups():
    # Unbalanced groups
    y = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    groups = [0, 0, 0, 0, 2, 2, 3, 3]
    weights, new_groups = calculate_sample_weight(
        y=y,
        groups=groups,
        weight_loss_by_class=False,  # Disabled
        weight_loss_by_groups=True,
        weight_per_split=False,
        weight_splits_equally=False,
    )

    assert_array_almost_equal(
        weights,
        [
            0.6666667,
            0.6666667,
            0.6666667,
            0.6666667,
            1.3333333,
            1.3333333,
            1.3333333,
            1.3333333,
        ],
    )
    # Test each group has same sum of weights
    _check_same_weights(new_groups, weights)
    assert_array_almost_equal(
        new_groups, groups  # No splits in the original groups so no difference
    )


def test_weighting_unbalanced_both():
    # Both unbalanced and a group with both labels
    y = np.asarray([0, 0, 0, 0, 0, 1, 1, 1])
    groups = [0, 0, 0, 0, 1, 1, 2, 2]

    # Step-wise
    unique_groups, group_weights = calculate_weights(v=groups)
    assert_array_almost_equal(unique_groups, [0, 1, 2])
    assert_array_almost_equal(
        group_weights,
        [
            (1 / 4) / (1 / 4 + 2 / 2) * 3,
            (1 / 2) / (1 / 4 + 2 / 2) * 3,
            (1 / 2) / (1 / 4 + 2 / 2) * 3,
        ],
    )
    sample_weights = np.array(
        [group_weights[int(g)] for g in groups]
    )  # Only works due to the group being indices 0-2

    print(sample_weights)

    unique_labels, label_weights = calculate_weights(v=y, weights=sample_weights)
    assert_array_almost_equal(unique_labels, [0, 1])
    print(label_weights)
    assert_array_almost_equal(
        label_weights,
        [
            (1 / sample_weights[y == 0].sum())
            / (1 / sample_weights[y == 0].sum() + 1 / sample_weights[y == 1].sum())
            * 2,
            (1 / sample_weights[y == 1].sum())
            / (1 / sample_weights[y == 0].sum() + 1 / sample_weights[y == 1].sum())
            * 2,
        ],
    )

    weights, new_groups = calculate_sample_weight(
        y=y,
        groups=groups,
        weight_loss_by_class=True,
        weight_loss_by_groups=True,
        weight_per_split=False,
        weight_splits_equally=False,
    )

    # In this case, the sums of the group weightings are the same
    # for both labels, and so, the weights should be 1 for both labels
    # Hence, the group weights are the results after both balancings
    # print(sample_weights[y == 0].sum(), sample_weights[y == 1].sum())
    assert_array_almost_equal(
        weights,
        [
            0.6666667,
            0.6666667,
            0.6666667,
            0.6666667,
            1.3333333,
            1.3333333,
            1.3333333,
            1.3333333,
        ],
    )
    # Test each group has same sum of weights
    _check_same_weights(new_groups, weights)
    assert_array_almost_equal(
        new_groups, groups  # No splits in the original groups so no difference
    )

    # If instead, the sum of the group weights are
    # different per label, the label weights would differ
    y = np.asarray([1, 0, 0, 0, 0, 1, 1, 0])
    assert_array_almost_equal(
        [sample_weights[y == 0].sum(), sample_weights[y == 1].sum()], [4.20, 3.00]
    )

    unique_labels, label_weights = calculate_weights(v=y, weights=sample_weights)
    assert_array_almost_equal(unique_labels, [0, 1])
    assert_array_almost_equal(
        label_weights,
        [
            (1 / sample_weights[y == 0].sum())
            / (1 / sample_weights[y == 0].sum() + 1 / sample_weights[y == 1].sum())
            * 2,
            (1 / sample_weights[y == 1].sum())
            / (1 / sample_weights[y == 0].sum() + 1 / sample_weights[y == 1].sum())
            * 2,
        ],
    )

    # Weighting sample weights by label weights
    sample_weights = sample_weights * np.array(
        [label_weights[int(l)] for l in y]
    )  # Only works due to the labels being indices 0-1

    assert_array_almost_equal(sample_weights, [0.7, 0.5, 0.5, 0.5, 1.0, 1.4, 1.4, 1.0])

    # Normalize to sum-to-length
    sample_weights = sample_weights / np.sum(sample_weights) * len(sample_weights)

    weights, new_groups = calculate_sample_weight(
        y=y,
        groups=groups,
        weight_loss_by_class=True,
        weight_loss_by_groups=True,
        weight_per_split=False,
        weight_splits_equally=False,
    )

    assert_array_almost_equal(weights, sample_weights)
    assert_array_almost_equal(
        new_groups, groups  # No splits in the original groups so no difference
    )


def test_weighting_unbalanced_both_per_split():
    # Both unbalanced and a group with both labels
    y = np.asarray(
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1]
    )
    orig_groups = [
        0,
        0,
        0,
        0,
        1,
        1,
        2,
        2,
        0,
        0,
        0,
        0,
        1,
        1,
        2,
        2,
        0,
        0,
        0,
        0,
        1,
        1,
        2,
        2,
    ]
    splits = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]
    groups = [
        f"{g}_|_{s}" for g, s in zip(orig_groups, splits)
    ]  # Add split identifiers

    weights, new_groups = calculate_sample_weight(
        y=y,
        groups=groups,
        weight_loss_by_class=True,
        weight_loss_by_groups=True,
        weight_per_split=True,
        weight_splits_equally=False,
    )

    # Test the **first split** is weighted as expected
    # The expected values are from `test_weighting_unbalanced_both()`
    assert_array_almost_equal(
        weights[:8],
        [
            0.666667,
            0.666667,
            0.666667,
            0.666667,
            1.3333333,
            1.3333333,
            1.3333333,
            1.3333333,
        ],
    )
    # Test each group has same sum of weights
    _check_same_weights(new_groups[:8], weights[:8])
    assert (
        new_groups == [str(g) for g in orig_groups]
    ).all()  # It removed split identifiers

    # If instead, the sum of the group weights are
    # different per label, the label weights would differ
    y = np.asarray(
        [1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0]
    )

    weights, new_groups = calculate_sample_weight(
        y=y,
        groups=groups,
        weight_loss_by_class=True,
        weight_loss_by_groups=True,
        weight_per_split=True,
        weight_splits_equally=False,
    )

    # The expected values are from `test_weighting_unbalanced_both()`
    assert_array_almost_equal(
        weights[:8], [0.8, 0.571429, 0.571429, 0.571429, 1.142857, 1.6, 1.6, 1.142857]
    )
    assert (
        new_groups == [str(g) for g in orig_groups]
    ).all()  # It removed split identifiersx

    # Differently sized splits that get equal overall weight
    # 8 0s, 7 1s, 5 2s
    splits = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    groups = [
        f"{g}_|_{s}" for g, s in zip(orig_groups, splits)
    ]  # Add split identifiers

    weights, new_groups = calculate_sample_weight(
        y=y[:-4],
        groups=groups,
        weight_loss_by_class=True,
        weight_loss_by_groups=True,
        weight_per_split=True,
        weight_splits_equally=True,  # Difference
    )

    # Test the **first split** is weighted as expected
    # The expected values are from `test_weighting_unbalanced_both()`
    assert_array_almost_equal(weights[:8].sum(), weights[8 : (8 + 7)].sum())
    assert_array_almost_equal(weights[:8].sum(), weights[(8 + 7) :].sum())

    ## One split has only one class
    # 8 0s, 7 1s, 5 2s
    # Only have one sample per subject (group)
    splits = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    groups = [
        f"{g}_|_{s}" for g, s in zip(range(len(splits)), splits)
    ]  # Add split identifiers

    y_ = y[:-4].copy()
    y_[np.asarray(splits) == 2] = 1
    weights, new_groups = calculate_sample_weight(
        y=y_,
        groups=groups,
        weight_loss_by_class=True,
        weight_loss_by_groups=True,
        weight_per_split=True,
        weight_splits_equally=False,
    )
    weights_no_per_split, _ = calculate_sample_weight(
        y=y_,
        groups=groups,
        weight_loss_by_class=True,
        weight_loss_by_groups=True,
        weight_per_split=False,
        weight_splits_equally=False,
    )

    # Check class sums per split (0 an 1, as 2 is only y_==1)
    # Check that the initial per-split makes a difference
    # compared to just global weighting
    combi_weights = [
        [
            weights[(np.asarray(splits) == 0) & (y_ == 0)].sum(),
            weights[(np.asarray(splits) == 0) & (y_ == 1)].sum(),
        ],
        [
            weights[(np.asarray(splits) == 1) & (y_ == 0)].sum(),
            weights[(np.asarray(splits) == 1) & (y_ == 1)].sum(),
        ],
        [weights[(y_ == 0)].sum(), weights[(y_ == 1)].sum()],
    ]
    print(combi_weights)
    assert_array_almost_equal(
        combi_weights,
        [
            [5.223880597014926, 3.0791788856304985],
            [4.776119402985076, 2.8152492668621703],
            [10.000000000000002, 9.999999999999998],
        ],
    )
    combi_weights_no_per_split = [
        [
            weights_no_per_split[(np.asarray(splits) == 0) & (y_ == 0)].sum(),
            weights_no_per_split[(np.asarray(splits) == 0) & (y_ == 1)].sum(),
        ],
        [
            weights_no_per_split[(np.asarray(splits) == 1) & (y_ == 0)].sum(),
            weights_no_per_split[(np.asarray(splits) == 1) & (y_ == 1)].sum(),
        ],
        [weights_no_per_split[(y_ == 0)].sum(), weights_no_per_split[(y_ == 1)].sum()],
    ]
    print(combi_weights_no_per_split)
    assert_array_almost_equal(
        combi_weights_no_per_split,
        [
            [5.555555555555557, 2.7272727272727284],
            [4.4444444444444455, 2.7272727272727284],
            [10.000000000000002, 10.000000000000005],
        ],
    )

    # Check global class sums
    assert_array_almost_equal(weights[(y_ == 0)].sum(), weights[(y_ == 1)].sum())

    ## Passing split weights
    split_weights = {"0": 1.0, "1": 1.0, "2": 0.1}

    weights, new_groups = calculate_sample_weight(
        y=y_,
        groups=groups,
        split_weights=split_weights,
        weight_loss_by_class=True,
        weight_loss_by_groups=True,
        weight_per_split=True,
        weight_splits_equally=False,
    )
    weights_no_per_split, _ = calculate_sample_weight(
        y=y_,
        groups=groups,
        split_weights=split_weights,
        weight_loss_by_class=True,
        weight_loss_by_groups=True,
        weight_per_split=False,
        weight_splits_equally=False,
    )

    # Check class sums per split (0 an 1, as 2 is only y_==1)
    # Check that the initial per-split makes a difference
    # compared to just global weighting
    combi_weights = [
        [
            weights[(np.asarray(splits) == 0) & (y_ == 0)].sum(),
            weights[(np.asarray(splits) == 0) & (y_ == 1)].sum(),
        ],
        [
            weights[(np.asarray(splits) == 1) & (y_ == 0)].sum(),
            weights[(np.asarray(splits) == 1) & (y_ == 1)].sum(),
        ],
        [weights[(y_ == 0)].sum(), weights[(y_ == 1)].sum()],
    ]
    print(combi_weights)
    # The within-split class balance should be closer together now that
    # the one-class split is weighted much lower overall!
    assert_array_almost_equal(
        combi_weights,
        [
            [5.223880597014928, 4.883720930232561],
            [4.776119402985076, 4.465116279069768],
            [10.000000000000004, 10.000000000000005],
        ],
    )
    combi_weights_no_per_split = [
        [
            weights_no_per_split[(np.asarray(splits) == 0) & (y_ == 0)].sum(),
            weights_no_per_split[(np.asarray(splits) == 0) & (y_ == 1)].sum(),
        ],
        [
            weights_no_per_split[(np.asarray(splits) == 1) & (y_ == 0)].sum(),
            weights_no_per_split[(np.asarray(splits) == 1) & (y_ == 1)].sum(),
        ],
        [weights_no_per_split[(y_ == 0)].sum(), weights_no_per_split[(y_ == 1)].sum()],
    ]
    print(combi_weights_no_per_split)
    # The global class imbalance with the split weighting
    assert_array_almost_equal(
        combi_weights_no_per_split,
        [
            [5.555555555555555, 4.615384615384618],
            [4.444444444444445, 4.615384615384618],
            [10.0, 10.000000000000004],
        ],
    )


def _check_same_weights(xs, weights):
    weight_sums = np.array([weights[xs == xi].sum() for xi in np.unique(xs)])
    weight_sums /= weight_sums.max()
    assert_array_almost_equal(weight_sums, [1.0 for _ in set(xs)])
