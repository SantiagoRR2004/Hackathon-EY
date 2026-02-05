import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import utils
import json
import os


def calculateCorrelation(col1: pd.Series, col2: pd.Series) -> float:
    """
    This function calculates the correlation between two columns.

    We use Procrustes similarity score. It can measure between
    matrices of different dimensions.

    The similarity score isn't actually called that, it is a
    value that can be obtained from using Procrustes analysis.

    Args:
        - col1 (pd.Series): The first column
        - col2 (pd.Series): The second column

    Returns:
        - (float): The correlation score between the two columns
    """
    # Stack into matrices
    X = np.vstack([np.atleast_1d(x).astype(float) for x in col1])
    Y = np.vstack([np.atleast_1d(y).astype(float) for y in col2])

    # Center the data
    X -= X.mean(axis=0)
    Y -= Y.mean(axis=0)

    # SVD of cross-covariance
    M = X.T @ Y
    U, S, Sv = np.linalg.svd(M, full_matrices=False)

    # 5) Procrustes similarity score (best metric)
    similarity = S.sum() / (np.linalg.norm(X, "fro") * np.linalg.norm(Y, "fro"))

    return similarity


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

    # Fill diagonal with NaN
    for i in range(len(heatmapDf)):
        heatmapDf.iloc[i, i] = np.nan

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
    cleanData = utils.loadCSV(os.path.join(dataFolder, "mental_healthCleaned.csv"))

    # Load the indexes
    with open(os.path.join(dataFolder, "indexes.json"), "r") as f:
        indexes = json.load(f)

    groups = [
        "Mental Health Support Index",
        "Workplace Stigma Index",
        "Organizational Openness Score",
    ]

    for g in groups:
        # Calculate the correlation matrix for the current group
        matrix = calculateCorrelationsMatrix(
            cleanData[[col for col in indexes[g] if col in cleanData.columns]]
        )

        data = matrix.to_numpy()
        labels = matrix.columns

        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(data, aspect="equal")

        # Add numbers inside the cells
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if i != j:  # Don't add numbers on the diagonal
                    ax.text(
                        j,
                        i,
                        f"{data[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="black",
                    )

        # Remove x-axis labels
        ax.set_xticks([])
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)

        plt.title(g)
        plt.colorbar(im)

        # Save the figure
        plt.savefig(
            os.path.join(dataFolder, f"{g.replace(' ', '')}.png"),
            bbox_inches="tight",
        )
        plt.close()


if __name__ == "__main__":
    indexCorrelation()
