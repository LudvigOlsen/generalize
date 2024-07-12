from numpy.testing import assert_array_almost_equal
from generalize.evaluate.prepare_inputs import aggregate_classification_predictions_by_group

def test_aggregate_classification_predictions_by_group():

    targets =       [1,1,1,         2,2,2,       1,1,1,       2,2,2,         1,1,1,       2,2,2,       1,1,1,       2,2,2]
    probabilities = [0.03,0.02,0.2, 0.7,0.8,0.9, 0.1,0.2,0.3, 0.85,0.9,0.95, 0.1,0.2,0.1, 0.1,0.2,0.3, 0.9,0.8,0.3, 0.7,0.71,0.72]
    # Note: Predictions are not meant to fit with probabilities in this test
    predictions =   [1,2,1,         2,2,2,       1,2,2,       2,2,1,         1,1,1,       2,2,2,       2,2,1,       1,1,1]
    groups =        [1,1,1,         2,2,2,       3,3,3,       4,4,4,         5,5,5,       6,6,6,       7,7,7,       8,8,8]

    new_targets, new_probabilities, new_predictions = aggregate_classification_predictions_by_group(
        targets=targets,
        probabilities=probabilities,
        predictions=predictions,
        groups=groups
    )

    assert_array_almost_equal(
        new_targets,
        [1,2,1,2,1,2,1,2],
    )
    assert_array_almost_equal(
        new_probabilities,
        [(0.03+0.02+0.2)/3, (0.7+0.8+0.9)/3, (0.1+0.2+0.3)/3, (0.85+0.9+0.95)/3, (0.1+0.2+0.1)/3, (0.1+0.2+0.3)/3, (0.9+0.8+0.3)/3, (0.7+0.71+0.72)/3],
    )
    assert_array_almost_equal(
        new_predictions,
        [1,2,2,2,1,2,2,1],
    )


# def test_aggregate_regression_predictions_by_group():

#     aggregate_regression_predictions_by_group(
#         targets: Optional[Union[list, np.ndarray]],
#         predictions: Optional[Union[list, np.ndarray]] = None,
#         groups: Optional[Union[list, np.ndarray]] = None,
#     )