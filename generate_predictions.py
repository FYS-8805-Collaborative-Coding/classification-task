# Example usage:
#   python generate_predictions.py --num-neighbors 7 --training-data train.csv --test-data test.csv --predictions data/predictions.csv

import click
import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist


def get_nearest_labels(row, labels, num_neighbors):
    nearest_labels = []
    for _distance, label in sorted(zip(row, labels)):
        nearest_labels.append(label)
        if len(nearest_labels) == num_neighbors:
            return nearest_labels


def nearest_neighbor_predictor(df_train, df_test, num_neighbors):
    test_positions = df_test[["x", "y"]].values
    train_positions = df_train[["x", "y"]].values
    train_labels = df_train["label"].values

    distance_matrix = cdist(test_positions, train_positions)

    predictions = []
    for row in distance_matrix:
        nearest_labels = get_nearest_labels(row, train_labels, num_neighbors)

        # this finds the most common nearest label by majority vote
        majority_index = np.bincount(nearest_labels).argmax()
        predictions.append(majority_index)

    return predictions


def fahrenheit_to_celsius(temp_f):
    """Converts temperature in Fahrenheit
    to Celsius.
    """
    temp_c = (temp_f - 32.0) * (5.0/9.0)
    return temp_c


# this is the test function
def test_fahrenheit_to_celsius():
    temp_c = fahrenheit_to_celsius(temp_f=100.0)
    expected_result = 37.777777
    # assert raises an error if the condition is not met
    assert abs(temp_c - expected_result) < 1.0e-6

@click.command()
@click.option(
    "--num-neighbors",
    type=int,
    required=True,
    help="Number of nearest neighbors in classifier. Should be an odd number to avoid ties.",
)
@click.option(
    "--training-data",
    type=str,
    required=True,
    help="We read the training data from this file.",
)
@click.option(
    "--test-data", type=str, required=True, help="We read the test data from this file."
)
@click.option(
    "--predictions",
    type=str,
    required=True,
    help="Predictions on test data are stored in this file.",
)
def main(num_neighbors, training_data, test_data, predictions):
    """
    Creates predictions on test data with a nearest neighbor classifier.
    """

    df_train = pd.read_csv(training_data)
    df_test = pd.read_csv(test_data)

    df_test["prediction"] = nearest_neighbor_predictor(df_train, df_test, num_neighbors)

    df_test.to_csv(predictions, index=False)
    print(f"Predictions saved to {predictions}")


if __name__ == "__main__":
    main()
