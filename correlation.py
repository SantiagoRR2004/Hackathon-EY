import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import textwrap
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

        1. We calculate the cross-covariance matrix between the two columns,
            this is done by centering and doing the dot product of the two matrices.

        2. We perform Singular Value Decomposition (SVD) on the cross-covariance.
            We only use the singular values (S) from the SVD, which represent the strength of the correlation.

        3. We sum the singular values and normalize by the product of the Frobenius norms of the original matrices.

    The similarity score ranges from 0 to 1 and
    permutations of the columns of either matrix do not change the score.
    See ./testCorrelation.py for experimental proof.

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
    U, S, V = np.linalg.svd(M, full_matrices=False)

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


def getTopCorrelatedPairs(matrix: pd.DataFrame, n: int = 5) -> list:
    """
    This function gets the top n pairs of questions with the highest correlation.

    Args:
        - matrix (pd.DataFrame): The correlation matrix
        - n (int): Number of top pairs to return

    Returns:
        - list: List of tuples (question1, question2, correlation_score)
    """
    pairs = []

    # Iterate through upper triangle of the matrix (excluding diagonal)
    for i in range(len(matrix.columns)):
        for j in range(i + 1, len(matrix.columns)):
            col1 = matrix.columns[i]
            col2 = matrix.columns[j]
            score = matrix.loc[col1, col2]

            if not np.isnan(score):
                pairs.append((col1, col2, score))

    # Sort by correlation score (descending)
    pairs.sort(key=lambda x: x[2], reverse=True)

    return pairs[:n]


def plotTopCorrelatedQuestions(
    topPairs: dict, dataFolder: str, title: str = None, versus: bool = True
) -> None:
    """
    Plot top correlated questions as horizontal bar charts.

    Args:
        - topPairs: dict mapping index names to lists of tuples.
            If versus is True, tuples are (q1, q2, score).
            If versus is False, tuples are (q, score).
        - dataFolder: path to save the generated images.
        - title: custom title for the chart. If None, a default title is generated.
        - versus: if True, labels show "q1 vs q2". If False, labels show a single question.
    """
    plt.style.use("ggplot")

    for indexName, pairs in topPairs.items():
        labels = []
        scores = []

        for i, entry in enumerate(pairs, 1):
            if versus:
                q1, q2, score = entry
                q1_f = textwrap.fill(q1, width=50)
                q2_f = textwrap.fill(q2, width=50)
                labels.append(f"{i}. {q1_f}\n--- vs ---\n{q2_f}")
            else:
                q, score = entry
                labels.append(f"{i}. {textwrap.fill(q, width=50)}")
            scores.append(score)

        fig, ax = plt.subplots(figsize=(12, 10))

        colors = plt.cm.viridis(np.linspace(0.8, 0.4, len(scores)))
        bars = ax.barh(labels, scores, color=colors, height=0.6)

        ax.bar_label(bars, fmt="%.3f", padding=8, fontweight="bold", fontsize=11)

        ax.tick_params(axis="y", labelsize=9)

        chartTitle = title if title is not None else f"Top 5 Correlations: {indexName}"
        ax.set_title(chartTitle, fontsize=15, pad=20, fontweight="bold")
        ax.set_xlim(0, max(scores) * 1.18)
        ax.invert_yaxis()
        plt.tight_layout()

        # Generate filename from index name
        safeFilename = f"Top5{indexName.replace(' ', '')}.png"

        # Save the figure
        plt.savefig(
            os.path.join(dataFolder, safeFilename),
            bbox_inches="tight",
            dpi=150,
            facecolor="white",
            edgecolor="none",
        )
        plt.close()


def indexCorrelation() -> dict:
    """
    This function calculates the correlation matrices for each index
    and generates visualizations including the top 5 correlated question pairs.

    Args:
        - None

    Returns:
        - dict: A dictionary containing the top 5 correlated pairs for each index
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

    # Dictionary to store top 5 pairs for each index
    allTopPairs = {}

    for g in groups:
        # Calculate the correlation matrix for the current group
        matrix = calculateCorrelationsMatrix(
            cleanData[[col for col in indexes[g] if col in cleanData.columns]]
        )

        # Get top 5 correlated pairs for this index
        topPairs = getTopCorrelatedPairs(matrix, n=5)
        allTopPairs[g] = topPairs

        # Print the top 5 correlations for this index
        utils.printSeparator(g)
        printable = []

        for i, (q1, q2, score) in enumerate(topPairs, 1):
            printable.append(f"{i}. Correlation: {score:.4f}\n\tQ1: {q1}\n\tQ2: {q2}")

        print("\n" + "\n\n".join(printable) + "\n")

        # Graph the correlation matrix for this index
        utils.graphCorrelationMatrix(matrix, g)

    # Generate the combined visualization for top 5 correlations by index
    plotTopCorrelatedQuestions(allTopPairs, dataFolder)

    return allTopPairs


if __name__ == "__main__":
    indexCorrelation()
