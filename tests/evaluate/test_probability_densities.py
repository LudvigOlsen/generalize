import pytest
import pandas as pd
import numpy as np
from generalize.evaluate.probability_densities import (
    ProbabilityDensities,
)

# TODO: These tests were generated and do not cover
# all aspects of the class. Improve the testing.


# Fixtures to create a simple DataFrame for testing
@pytest.fixture
def simple_df():
    np.random.seed(42)
    return pd.DataFrame(
        {"probability": np.random.rand(200), "class": np.random.choice(["A", "B"], 200)}
    )


@pytest.fixture
def grouped_df():
    np.random.seed(42)
    return pd.DataFrame(
        {
            "group": np.repeat(["G1", "G2"], 100),
            "probability": np.random.rand(200),
            "class": np.random.choice(["A", "B"], 200),
        }
    )


# Test cases for ProbabilityDensities


def test_probd_initialization_without_group(simple_df):
    """Test the initialization without groups"""
    pd_object = ProbabilityDensities().calculate_densities(simple_df, "probability", "class")
    assert pd_object.densities is not None
    assert "expected accuracy" in pd_object.densities.columns


def test_probd_initialization_with_group(grouped_df):
    """Test the initialization with groups"""
    pd_object = ProbabilityDensities().calculate_densities(grouped_df, "probability", "class", ["group"])
    assert pd_object.densities is not None
    assert "expected accuracy" in pd_object.densities.columns


def test_probd_compute_density(simple_df):
    """Test the density calculation function"""
    densities = ProbabilityDensities._compute_density(simple_df, "class", "probability")
    assert not densities.empty
    assert set(densities["class"]) == {"A", "B"}
    assert all(0 <= densities["probability"]) and all(densities["probability"] <= 1)
    assert all(0 <= densities["density"]) and all(densities["density"] <= 1)


def test_probd_calculate_expected_accuracy(simple_df):
    """Test the calculation of expected accuracy"""
    pd_object = ProbabilityDensities().calculate_densities(simple_df, "probability", "class")
    result_df = pd_object.densities
    assert "expected accuracy" in result_df.columns
    assert not result_df[["expected accuracy"]].isnull().values.any()
    assert all(0 <= result_df["expected accuracy"]) and all(
        result_df["expected accuracy"] <= 1
    )


def test_probd_get_expected_accuracy(simple_df):
    """Test getting expected accuracy for a new probability"""
    pd_object = ProbabilityDensities().calculate_densities(simple_df, "probability", "class")
    new_probability = 0.5
    expected_accuracies = pd_object.get_expected_accuracy(new_probability)
    assert "A" in expected_accuracies
    assert "B" in expected_accuracies
    assert 0 <= expected_accuracies["A"] <= 1
    assert 0 <= expected_accuracies["B"] <= 1


def test_probd_invalid_probability(simple_df):
    """Test that invalid probability raises a ValueError"""
    pd_object = ProbabilityDensities().calculate_densities(simple_df, "probability", "class")
    with pytest.raises(ValueError):
        pd_object.get_expected_accuracy(-0.1)
    with pytest.raises(ValueError):
        pd_object.get_expected_accuracy(1.1)


def test_probd_empty_dataframe():
    """Test handling of an empty DataFrame"""
    empty_df = pd.DataFrame(columns=["probability", "class"])
    with pytest.raises(ValueError):
        ProbabilityDensities().calculate_densities(empty_df, "probability", "class")


def test_probd_multiple_group_columns():
    """Test handling of multiple group columns with data"""
    np.random.seed(42)

    df = pd.DataFrame(
        {
            "group1": np.repeat(["G1", "G2"], 50),
            "group2": np.tile(["A", "B"], 50),
            "probability": np.random.rand(100),
            "class": np.random.choice(["A", "B"], 100),
        }
    )

    pd_object = ProbabilityDensities().calculate_densities(df, "probability", "class", ["group1", "group2"])

    # Ensure densities are calculated correctly for multiple group columns
    assert not pd_object.densities.empty
    assert set(pd_object.densities["class"]) == {"A", "B"}
    assert all(0 <= pd_object.densities["probability"]) and all(
        pd_object.densities["probability"] <= 1
    )
    assert all(0 <= pd_object.densities["density"]) and all(
        pd_object.densities["density"] <= 1
    )

    # Check if expected accuracy column is present
    assert "expected accuracy" in pd_object.densities.columns


def test_probd_save_and_from_file(tmp_path):
    """Test the save and from_file methods of ProbabilityDensities"""

    np.random.seed(42)
    # Create example data
    df = pd.DataFrame(
        {
            "group": np.repeat(["G1", "G2"], 100),
            "probability": np.random.rand(200),
            "class": np.random.choice(["A", "B"], 200),
        }
    )

    # Initialize ProbabilityDensities object
    pd_object = ProbabilityDensities().calculate_densities(df, "probability", "class", ["group"])

    # Define the path for saving the CSV
    csv_path = tmp_path / "densities.csv"

    # Save the densities to CSV
    pd_object.save(csv_path)

    # Test get_expected_accuracy with the loaded object
    new_probability = 0.5

    original_exp_acc = pd_object.get_expected_accuracy(new_probability)

    loaded_pd_object = pd_object.from_file(csv_path)

    loaded_exp_acc = loaded_pd_object.get_expected_accuracy(new_probability)

    original_exp_acc == loaded_exp_acc
