from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import json
import os


def calculateCorrelation(col1: pd.Series, col2: pd.Series) -> float:
    """
    This function calculates the correlation between two columns.

    Because some columns can be vectors, we use PCA to reduce the dimensionality
    of the bigger vector to the smaller one, and then calculate the cosine similarity.

    Args:
        - col1 (pd.Series): The first column
        - col2 (pd.Series): The second column

    Returns:
        - (float): The correlation score between the two columns
    """
    # Change around so vectors col1 < col2
    if len(col1[0]) > len(col2[0]):
        col1, col2 = col2, col1

    nCompo = len(col1[0])

    pca = PCA(n_components=nCompo)
    transformed = pca.fit_transform(
        np.hstack(
            [
                np.vstack(col2.to_numpy()),
            ]
        )
    )

    # Make sure the lengths match
    assert len(col1[0]) == len(transformed[0])

    # Cosine similarity
    score = cosine_similarity(np.vstack(col1), transformed).flatten().mean()

    return score


def calculateCorrelationsMatrix(data: pd.DataFrame) -> pd.DataFrame:
    """
    This function calculates the correlation matrix of the given data.

    Args:
        - data: The data to calculate the correlations for

    Returns:
        - A dictionary with the correlation matrix
    """
    heatmapDf = pd.DataFrame(np.nan, index=data.columns, columns=data.columns)

    # Iterate across all pairs of columns
    for i in range(len(data.columns)):
        for j in range(i + 1, len(data.columns)):
            col1 = data.columns[i]
            col2 = data.columns[j]

            # Calculate correlation
            score = calculateCorrelation(data[col1], data[col2])

            # Symmetric matrix
            heatmapDf.loc[col1, col2] = score
            heatmapDf.loc[col2, col1] = score

    # Fill diagonal with 1
    for i in range(len(heatmapDf)):
        heatmapDf.iloc[i, i] = 1

    return heatmapDf


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
        calculateCorrelationsMatrix(
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
