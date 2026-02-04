import pandas as pd
import numpy as np
import json
import os


def calculateCorrelations(data: pd.DataFrame) -> dict:
    """
    This function calculates the correlation matrix of the given data.

    Args:
        - data: The data to calculate the correlations for

    Returns:
        - A dictionary with the correlation matrix
    """
    # Iterate across all pairs of columns
    for i in range(len(data.columns)):
        for j in range(i + 1, len(data.columns)):
            col1 = data.columns[i]
            col2 = data.columns[j]

            print(col1, col2)


def indexCorrelation() -> None:
    """

    Args:
        - None

    Returns:
        - None
    """
    # Get the paths
    currentDirectory = os.path.dirname(os.path.abspath(__file__))
    dataFolder = os.path.join(currentDirectory, "data")

    # Load the data
    cleanData = pd.read_csv(os.path.join(dataFolder, "mental_healthCleaned.csv"))

    # Fix vectors
    for c in cleanData.columns:
        if (
            pd.api.types.is_string_dtype(cleanData[c])
            and cleanData[c][0].startswith("[")
            and cleanData[c][0].endswith("]")
        ):
            cleanData[c] = cleanData[c].apply(
                lambda x: np.fromstring(x.strip("[]"), sep=" ")
            )

    # Load the indexes
    with open(os.path.join(dataFolder, "indexes.json"), "r") as f:
        indexes = json.load(f)

    # Mental Health Support Index
    print(
        calculateCorrelations(
            cleanData[
                [
                    col
                    for col in indexes["Mental Health Support Index"]
                    if col in cleanData.columns
                ]
            ]
        )
    )


if __name__ == "__main__":
    indexCorrelation()
