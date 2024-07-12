

import numpy as np
from nattrs import nested_getattr

from generalize.evaluate.confusion_matrices import ConfusionMatrices

# TODO Also test multiclass!


def test_confusion_matrices_set_get():

    seed = 15
    np.random.seed(seed)

    conf_coll = ConfusionMatrices(
        class_roles={'negative': 'control', 'positive': 'case'},
        count_names=np.array([['TN', 'FP'], ['FN', 'TP']])
    )

    for rep in range(3):
        for split in range(2):
            for thresh in ["a", "b"]:
                conf_coll.add(
                    path=f"rep.{rep}.split.{split}.threshold.{thresh}",
                    matrix=np.random.choice(range(10), size=(2, 2))
                )

    print(conf_coll._matrices)

    conf_coll_matrices = {
        'rep': {
            '0': {'split': {
                '0': {'threshold': {
                    'a': np.array([[8, 5], [5, 7]]),
                    'b': np.array([[0, 7], [5, 6]])}},
                '1': {'threshold': {
                    'a': np.array([[1, 7], [0, 4]]),
                    'b': np.array([[9, 7], [5, 3]])}}}},
            '1': {'split': {
                '0': {'threshold': {
                    'a': np.array([[6, 8], [2, 1]]),
                    'b': np.array([[1, 0], [5, 2]])}},
                '1': {'threshold': {
                    'a': np.array([[2, 1], [8, 5]]),
                    'b': np.array([[6, 9], [2, 8]])}}}},
            '2': {'split': {
                '0': {'threshold': {
                    'a': np.array([[6, 8], [8, 3]]),
                    'b': np.array([[4, 7], [2, 0]])}},
                '1': {'threshold': {
                    'a': np.array([[5, 7], [3, 8]]),
                    'b': np.array([[5, 3], [1, 0]])
                }}
            }}
        }
    }

    # Check all elements
    assert (conf_coll._matrices["rep"].keys() ==
            conf_coll_matrices["rep"].keys())
    for rep in range(3):
        assert (nested_getattr(conf_coll._matrices, f"rep.{rep}").keys() ==
                nested_getattr(conf_coll_matrices, f"rep.{rep}").keys())
        assert (nested_getattr(conf_coll._matrices, f"rep.{rep}.split").keys() ==
                nested_getattr(conf_coll_matrices, f"rep.{rep}.split").keys())
        for split in range(2):
            assert (nested_getattr(conf_coll._matrices, f"rep.{rep}.split.{split}").keys() ==
                    nested_getattr(conf_coll_matrices, f"rep.{rep}.split.{split}").keys())
            assert (nested_getattr(conf_coll._matrices, f"rep.{rep}.split.{split}.threshold").keys() ==
                    nested_getattr(conf_coll_matrices, f"rep.{rep}.split.{split}.threshold").keys())
            for thresh in ["a", "b"]:
                assert (nested_getattr(conf_coll._matrices, f"rep.{rep}.split.{split}.threshold.{thresh}") ==
                        nested_getattr(conf_coll_matrices, f"rep.{rep}.split.{split}.threshold.{thresh}")).all()
                # Test getter
                assert (nested_getattr(conf_coll_matrices, f"rep.{rep}.split.{split}.threshold.{thresh}") ==
                        conf_coll.get(path=f"rep.{rep}.split.{split}.threshold.{thresh}")).all()

    # print(conf_coll.paths)
    assert conf_coll.paths == [
        'rep.0.split.0.threshold.a', 'rep.0.split.0.threshold.b',
        'rep.0.split.1.threshold.a', 'rep.0.split.1.threshold.b',
        'rep.1.split.0.threshold.a', 'rep.1.split.0.threshold.b',
        'rep.1.split.1.threshold.a', 'rep.1.split.1.threshold.b',
        'rep.2.split.0.threshold.a', 'rep.2.split.0.threshold.b',
        'rep.2.split.1.threshold.a', 'rep.2.split.1.threshold.b'
    ]


def test_confusion_matrices_save_load_binary(tmp_path):

    seed = 15
    np.random.seed(seed)

    conf_coll = ConfusionMatrices(
        class_roles={'negative': 'control', 'positive': 'case'},
        count_names=np.array([['TN', 'FP'], ['FN', 'TP']]),
        note="A note!"
    )

    for rep in range(2):
        for split in range(2):
            conf_coll.add(
                path=f"rep.{rep}.split.{split}",
                matrix=np.random.choice(range(10), size=(2, 2))
            )

    conf_coll.save(file_path=tmp_path / "tmp_conf_mat.json")
    loaded_conf_coll = ConfusionMatrices.load(
        file_path=tmp_path / "tmp_conf_mat.json"
    )

    print(loaded_conf_coll._matrices)

    # Check non-matrix attributes
    assert conf_coll.classes == loaded_conf_coll.classes
    assert conf_coll.class_roles == loaded_conf_coll.class_roles
    assert (conf_coll.count_names == loaded_conf_coll.count_names).all()
    assert conf_coll.paths == loaded_conf_coll.paths
    assert conf_coll.note == loaded_conf_coll.note

    # Check all matrix elements
    assert (conf_coll._matrices["rep"].keys() ==
            loaded_conf_coll._matrices["rep"].keys())
    for rep in range(2):
        assert (nested_getattr(conf_coll._matrices, f"rep.{rep}").keys() ==
                nested_getattr(loaded_conf_coll._matrices, f"rep.{rep}").keys())
        assert (nested_getattr(conf_coll._matrices, f"rep.{rep}.split").keys() ==
                nested_getattr(loaded_conf_coll._matrices, f"rep.{rep}.split").keys())
        for split in range(2):
            assert (nested_getattr(conf_coll._matrices, f"rep.{rep}.split.{split}") ==
                    nested_getattr(loaded_conf_coll._matrices, f"rep.{rep}.split.{split}")).all()

    print(loaded_conf_coll)


def test_confusion_matrices_save_load_multiclass(tmp_path):

    seed = 15
    np.random.seed(seed)

    conf_coll = ConfusionMatrices(
        classes=np.array([1, 2, 3]).tolist(),
        count_names=np.array([
            ['0==0', '0==1', '0==2'],
            ['1==0', '1==1', '1==2'],
            ['2==0', '2==1', '2==2']
        ]),
        note="A note!"
    )

    for rep in range(2):
        for split in range(2):
            conf_coll.add(
                path=f"rep.{rep}.split.{split}",
                matrix=np.random.choice(range(10), size=(3, 3))
            )

    conf_coll.save(file_path=tmp_path / "tmp_conf_mat.json")
    loaded_conf_coll = ConfusionMatrices.load(
        file_path=tmp_path / "tmp_conf_mat.json"
    )

    print(loaded_conf_coll._matrices)

    # Check non-matrix attributes
    assert conf_coll.classes == loaded_conf_coll.classes
    assert conf_coll.class_roles is None and loaded_conf_coll.class_roles is None
    assert (conf_coll.count_names == loaded_conf_coll.count_names).all()
    assert conf_coll.paths == loaded_conf_coll.paths
    assert conf_coll.note == loaded_conf_coll.note

    # Check all matrix elements
    assert (conf_coll._matrices["rep"].keys() ==
            loaded_conf_coll._matrices["rep"].keys())
    for rep in range(2):
        assert (nested_getattr(conf_coll._matrices, f"rep.{rep}").keys() ==
                nested_getattr(loaded_conf_coll._matrices, f"rep.{rep}").keys())
        assert (nested_getattr(conf_coll._matrices, f"rep.{rep}.split").keys() ==
                nested_getattr(loaded_conf_coll._matrices, f"rep.{rep}.split").keys())
        for split in range(2):
            assert (nested_getattr(conf_coll._matrices, f"rep.{rep}.split.{split}") ==
                    nested_getattr(loaded_conf_coll._matrices, f"rep.{rep}.split.{split}")).all()

    print(loaded_conf_coll)


def test_confusion_matrices_operators():

    seed = 15
    np.random.seed(seed)

    conf_coll = ConfusionMatrices(
        class_roles={'negative': 'control', 'positive': 'case'},
        count_names=np.array([['TN', 'FP'], ['FN', 'TP']]),
        note="A note!"
    )

    for rep in range(2):
        for split in range(2):
            conf_coll.add(
                path=f"rep.{rep}.split.{split}",
                matrix=np.random.choice(range(10), size=(2, 2))
            )

    # Add integer to all counts
    conf_coll += 3
    conf_coll /= 2
    print(conf_coll)

    conf_coll //= 1
    print(conf_coll)

    conf_coll_2 = conf_coll.__copy__()

    conf_coll /= conf_coll_2
    print(conf_coll)
    assert False, "Add equality tests"
