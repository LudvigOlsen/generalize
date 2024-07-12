import numpy as np
from string import ascii_lowercase
from nattrs import nested_getattr

from generalize.evaluate.evaluate_repetitions import evaluate_repetitions

# TODO Test regression and multiclass!

###############################
#### Binary classification ####
###############################


def test_evaluate_repetitions_with_changing_splits_binary_clf(
    create_splits_fn, create_y_fn
):
    # TODO Convert to fixture
    seed = 15
    np.random.seed(seed)

    # NOTE: These are *regression tests*

    num_samples = 30
    num_splits = 3
    num_reps = 2

    targets = create_y_fn(num_samples=num_samples)
    predictions_list = [
        create_y_fn(num_samples, digitize=False) for _ in range(num_reps)
    ]
    splits_list = [
        create_splits_fn(
            num_samples=num_samples,
            num_splits=num_splits,
            id_names=list(ascii_lowercase)[:num_splits],
        )
        for _ in range(num_reps)
    ]

    eval = evaluate_repetitions(
        predictions_list=predictions_list,
        targets=targets,
        task="binary_classification",
        positive=1,
        splits_list=splits_list,
        thresholds=None,
        target_labels={0: "control", 1: "case"},
        identifier_cols_dict=None,
        eval_idx_colname="Repetition",
        split_id_colname="Fold",
    )

    # print(eval)

    # TODO
    # assert False, "Lacking equality tests"

    # With groups
    seed = 15
    groups = np.random.choice(10, size=(30))
    targets = groups.copy()
    target_by_group = np.random.choice(2, size=(10))
    for group_idx, group in enumerate(np.unique(groups)):
        targets[groups == group] = target_by_group[group_idx]

    eval = evaluate_repetitions(
        predictions_list=predictions_list,
        targets=targets,
        groups=groups,
        task="binary_classification",
        positive=1,
        splits_list=splits_list,
        thresholds=None,
        target_labels={0: "control", 1: "case"},
        identifier_cols_dict=None,
        eval_idx_colname="Repetition",
        split_id_colname="Fold",
    )

    # print(eval)

    cm = nested_getattr(eval, "Evaluations.Confusion Matrices").get(
        "Repetition.0.Threshold Version.0_5 Threshold.Fold.0"
    )
    assert (cm == np.asarray([[5, 0], [0, 1]])).all()
    # Note: The splits and groups means there will
    # be <= 10 counts per fold
    # assert cm.sum() == len(np.unique(groups))

    assert False, "Lacking equality tests"


def test_evaluate_repetitions_with_static_splits_binary_clf(
    create_splits_fn, create_y_fn
):
    # TODO Convert to fixture
    seed = 15
    np.random.seed(seed)

    # NOTE: These are *regression tests*

    num_samples = 30
    num_splits = 3
    num_reps = 2

    targets = create_y_fn(num_samples=num_samples)
    predictions_list = [
        create_y_fn(num_samples, digitize=False) for _ in range(num_reps)
    ]
    single_split = create_splits_fn(
        num_samples=num_samples,
        num_splits=num_splits,
        id_names=list(ascii_lowercase)[:num_splits],
    )
    splits_list = [single_split for _ in range(num_reps)]

    eval = evaluate_repetitions(
        predictions_list=predictions_list,
        targets=targets,
        task="binary_classification",
        positive=1,
        splits_list=splits_list,
        summarize_splits=True,
        thresholds=None,
        target_labels={0: "control", 1: "case"},
        identifier_cols_dict=None,
        eval_idx_colname="Repetition",
        split_id_colname="Fold",
    )

    print(eval)

    assert False, "Lacking equality tests"


def test_evaluate_repetitions_with_no_splits_binary_clf(create_y_fn):
    # TODO Convert to fixture
    seed = 15
    np.random.seed(seed)

    # NOTE: These are *regression tests*

    num_samples = 30
    num_reps = 2

    targets = create_y_fn(num_samples=num_samples)
    predictions_list = [
        create_y_fn(num_samples, digitize=False) for _ in range(num_reps)
    ]

    eval = evaluate_repetitions(
        predictions_list=predictions_list,
        targets=targets,
        task="binary_classification",
        positive=1,
        splits_list=None,
        summarize_splits=False,
        thresholds=None,
        target_labels={0: "control", 1: "case"},
        identifier_cols_dict=None,
        eval_idx_colname="Repetition",
        split_id_colname="Fold",
    )

    print(eval.keys())
    print(eval)

    assert False, "Lacking equality tests"


###################################
#### Multiclass classification ####
###################################


def test_evaluate_repetitions_with_changing_splits_mc_clf(
    create_splits_fn, create_y_fn
):
    # TODO Convert to fixture
    seed = 15
    np.random.seed(seed)

    # NOTE: These are *regression tests*

    num_samples = 30
    num_splits = 3
    num_reps = 2

    targets = create_y_fn(num_samples=num_samples, num_classes=3)
    predictions_list = [
        create_y_fn(num_samples, digitize=False, num_classes=3) for _ in range(num_reps)
    ]
    splits_list = [
        create_splits_fn(
            num_samples=num_samples,
            num_splits=num_splits,
            id_names=list(ascii_lowercase)[:num_splits],
        )
        for _ in range(num_reps)
    ]

    eval = evaluate_repetitions(
        predictions_list=predictions_list,
        targets=targets,
        task="multiclass_classification",
        splits_list=splits_list,
        thresholds=None,
        target_labels={0: "control", 1: "case_1", 2: "case_2"},
        identifier_cols_dict=None,
        eval_idx_colname="Repetition",
        split_id_colname="Fold",
    )

    print(eval.keys())
    print(eval)

    assert False, "Lacking equality tests"


def test_evaluate_repetitions_with_static_splits_mc_clf(create_splits_fn, create_y_fn):
    # TODO Convert to fixture
    seed = 15
    np.random.seed(seed)

    # NOTE: These are *regression tests*

    num_samples = 30
    num_splits = 3
    num_reps = 2

    targets = create_y_fn(num_samples=num_samples, num_classes=3)
    predictions_list = [
        create_y_fn(num_samples, digitize=False, num_classes=3) for _ in range(num_reps)
    ]
    single_split = create_splits_fn(
        num_samples=num_samples,
        num_splits=num_splits,
        id_names=list(ascii_lowercase)[:num_splits],
    )
    splits_list = [single_split for _ in range(num_reps)]

    eval = evaluate_repetitions(
        predictions_list=predictions_list,
        targets=targets,
        task="multiclass_classification",
        splits_list=splits_list,
        summarize_splits=True,
        thresholds=None,
        target_labels={0: "control", 1: "case_1", 2: "case_2"},
        identifier_cols_dict=None,
        eval_idx_colname="Repetition",
        split_id_colname="Fold",
    )

    print(eval.keys())
    print(eval)

    assert False, "Lacking equality tests"


def test_evaluate_repetitions_with_no_splits_mc_clf(create_y_fn):
    # TODO Convert to fixture
    seed = 15
    np.random.seed(seed)

    # NOTE: These are *regression tests*

    num_samples = 30
    num_reps = 2

    targets = create_y_fn(num_samples=num_samples, num_classes=3)
    predictions_list = [
        create_y_fn(num_samples, digitize=False, num_classes=3) for _ in range(num_reps)
    ]

    eval = evaluate_repetitions(
        predictions_list=predictions_list,
        targets=targets,
        task="multiclass_classification",
        splits_list=None,
        summarize_splits=False,
        thresholds=None,
        target_labels={0: "control", 1: "case_1", 2: "case_2"},
        identifier_cols_dict=None,
        eval_idx_colname="Repetition",
        split_id_colname="Fold",
    )

    print(eval.keys())
    print(eval)

    assert False, "Lacking equality tests"
