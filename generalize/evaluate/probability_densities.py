import pathlib
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d


class ProbabilityDensities:
    def __init__(self) -> None:
        """
        Calculate densities of probabilities per class
        to check the expected accuracy of a predicted
        probability when the probabilistic classifier may
        not be perfectly calibrated.

        NOTE: Developed for binomial models.

        Example
        -------
        >>> prob_densities = ProbabilityDensities()
        >>> prob_densities.calculate_densities(df=df, probability_col=probability_col, ...)
        >>> prob_densities.get_expected_accuracy(new_probability=0.76)
        {'Class A': array(0.54119503), 'Class B': array(0.45880497)}
        """
        pass

    def calculate_densities(
        self,
        df: pd.DataFrame,
        probability_col: str,
        target_col: str,
        group_cols: Optional[Union[str, List[str]]] = None,
    ):
        """
        Calculate the probability densities per class.

        Parameters
        ----------
        df
            Data frame with predicted probabilities, target classes,
            and optionally some grouping columns (NOTE: densities are calculated
            per group and averaged).
        probability_col
            Name of column in `df` with predicted probabilities.
        target_col
            Name of column in `df` with target classes.
        group_cols
            Name(s) of column(s) in `df` with groupings.
            The densities are calculated per group and averaged.
            To get separate densities per group, use multiple
            instances of `ProbabilityDensities` instead.

        Returns
        -------
        self
        """
        self.densities = ProbabilityDensities._calculate_densities(
            df=df.copy(),
            target_col=target_col,
            probability_col=probability_col,
            group_cols=group_cols,
        )
        return self

    def save(self, path: Union[pathlib.Path, str]):
        """
        Save the densities data frame to the disk.

        You can recreate the `ProbabilityDensities` object
        by using `.from_file()`

        Parameters
        ----------
        path
            A path to save the file at.
            Must end with ".csv".

        Returns
        -------
        self
        """
        path = ProbabilityDensities._check_csv_path(path)
        self.densities.to_csv(path, header=True, index=False)
        return self

    @staticmethod
    def from_file(path: Union[pathlib.Path, str]):
        """
        Load a densities csv file from disk and create a
        new `ProbabilityDensities` instance.

        Parameters
        ----------
        path
            A path to load the file from.
            Must end with ".csv".

        Returns
        -------
        `ProbabilityDensities`
            Instance of `ProbabilityDensities` with the
            loaded densities assigned.
        """
        path = ProbabilityDensities._check_csv_path(path)
        probd = ProbabilityDensities()
        probd.densities = pd.read_csv(path)
        return probd

    @staticmethod
    def _check_csv_path(path: Union[pathlib.Path, str]) -> pathlib.Path:
        path = pathlib.Path(path)
        if not path.suffix == ".csv":
            raise ValueError(f"`path` must have the extension '.csv'. Got: {path}")
        return path

    @staticmethod
    def _calculate_densities(df, target_col, probability_col, group_cols=None):
        if df.empty:
            raise ValueError("`df` was empty.")

        # Convert targets to strings
        df[target_col] = df[target_col].astype("string")

        if group_cols is not None and group_cols:
            # Calculate densities per group
            group_densities = []
            for names, group in df.groupby(group_cols):
                density_data = ProbabilityDensities._compute_density(
                    group, target_col, probability_col
                )
                # TODO: Check it works with multiple group columns
                density_data.loc[:, group_cols] = names
                group_densities.append(density_data)

            # Combine all group densities into a single DataFrame
            density_df = pd.concat(group_densities)

            # Average the densities across groups
            avg_density_df = (
                density_df.groupby(["probability", "class"])
                .agg({"density": "mean"})
                .reset_index()
            )
        else:
            # Calculate densities for the entire dataset
            avg_density_df = ProbabilityDensities._compute_density(
                df, target_col, probability_col
            )

        # Calculate expected accuracy based on average densities
        avg_density_df = ProbabilityDensities._calculate_expected_accuracy(
            avg_density_df
        )

        return avg_density_df

    @staticmethod
    def _compute_density(df, target_col, probability_col):
        density_data = []
        classes = df[target_col].unique()

        for cls in classes:
            subset = df[df[target_col] == cls]
            if subset.empty:
                continue

            kde = gaussian_kde(subset[probability_col], bw_method="scott")
            prob_values = np.linspace(0, 1, 512)  # 2^9
            density = kde(prob_values)
            density /= density.max()  # Normalize to have a maximum of 1

            density_data.append(
                pd.DataFrame(
                    {"probability": prob_values, "density": density, "class": cls}
                )
            )

        return pd.concat(density_data)

    @staticmethod
    def _calculate_expected_accuracy(density_df):
        # Pivot table to calculate expected accuracy
        pivot_df = density_df.pivot(
            index="probability", columns="class", values="density"
        ).fillna(0)
        pivot_df["total_density"] = pivot_df.sum(axis=1)

        # Calculate expected accuracy as percentage of each class density
        for cls in pivot_df.columns.drop("total_density"):
            pivot_df[f"expected accuracy {cls}"] = (
                pivot_df[cls] / pivot_df["total_density"]
            )

        # Convert back to long format for easier plotting
        result_df = pivot_df.reset_index().melt(
            id_vars="probability",
            value_vars=[
                f"expected accuracy {cls}" for cls in density_df["class"].unique()
            ],
            var_name="ColumnName",
            value_name="expected accuracy",
        )
        result_df["class"] = result_df.ColumnName.apply(
            lambda s: s.replace("expected accuracy", "").strip()
        )
        del result_df["ColumnName"]

        return pd.merge(density_df, result_df, on=["class", "probability"], how="left")

    def get_expected_accuracy(self, new_probability: float) -> Dict[str, float]:
        """
        Calculate the expected accuracy for a new probability.

        Parameters
        ----------
        new_probability
            The new probability for which to calculate the expected accuracy.

        Returns
        -------
        dict
            A dictionary with classes as keys and the
            corresponding expected accuracies as values.
        """
        # Ensure the new probability is within the range [0, 1]
        if new_probability < 0 or new_probability > 1:
            raise ValueError("Probability must be between 0 and 1.")

        # Create a dictionary to store the expected accuracy for each class
        expected_accuracies = {}

        for name, group in self.densities.groupby("class"):
            # Interpolate expected accuracy for the new probability
            interp_func = interp1d(
                group["probability"],
                group["expected accuracy"],
                kind="linear",
                fill_value="extrapolate",
            )
            expected_accuracies[name] = interp_func(new_probability)

        return expected_accuracies


if __name__ == "__main__":
    # Example usage
    np.random.seed(1)

    # Multiple grouping columns
    df = pd.DataFrame(
        {
            "group1": np.repeat(["G1", "G2"], 50),
            "group2": np.tile(["A", "B"], 50),
            "probability": np.random.rand(100),
            "class": np.random.choice(["A", "B"], 100),
        }
    )
    prob_densities = ProbabilityDensities(
        df,
        probability_col="probability",
        target_col="class",
        group_cols="group1",
    )

    print(prob_densities.densities)

    # Example usage
    # Assuming result_df is already computed from the previous example
    new_probability = 0.75
    expected_accuracies = prob_densities.get_expected_accuracy(
        new_probability=new_probability
    )
    print(
        f"Expected Accuracies for probability {new_probability}: {expected_accuracies}"
    )

    prob_densities = ProbabilityDensities(
        df,
        probability_col="probability",
        target_col="class",
        group_cols=["group1", "group2"],
    )

    print(prob_densities.densities)

    # Example usage
    # Assuming result_df is already computed from the previous example
    new_probability = 0.75
    expected_accuracies = prob_densities.get_expected_accuracy(
        new_probability=new_probability
    )
    print(
        f"Expected Accuracies for probability {new_probability}: {expected_accuracies}"
    )

    print(prob_densities.densities)
