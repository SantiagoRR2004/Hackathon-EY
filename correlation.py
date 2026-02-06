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


def plotTopCorrelatedQuestions(topPairs: dict, dataFolder: str) -> None:
    """
    This function creates separate visualizations for each index showing
    the top 5 correlated question pairs.

    Args:
        - topPairs (dict): Dictionary with index names as keys and list of top pairs as values
        - dataFolder (str): Path to the data folder

    Returns:
        - None
    """
    import textwrap

    # Professional color palette
    colorPalettes = {
        "Mental Health Support Index": [
            "#1a5f4a",
            "#238b6a",
            "#2cb67d",
            "#5dd9a3",
            "#a8f0d4",
        ],
        "Workplace Stigma Index": [
            "#1e3a5f",
            "#2563a8",
            "#3b82f6",
            "#60a5fa",
            "#93c5fd",
        ],
        "Organizational Openness Score": [
            "#7c2d12",
            "#b45309",
            "#d97706",
            "#f59e0b",
            "#fbbf24",
        ],
    }

    for indexName, pairs in topPairs.items():
        # Set style
        plt.style.use("seaborn-v0_8-whitegrid")

        # Create individual figure for each index
        fig, ax = plt.subplots(figsize=(16, 12))

        colors = colorPalettes.get(
            indexName, ["#4a5568", "#718096", "#a0aec0", "#cbd5e0", "#e2e8f0"]
        )

        # Prepare data with FULL text wrapped
        labels = []
        scores = []

        for i, (q1, q2, score) in enumerate(pairs):
            # Wrap text to fit nicely (no truncation)
            wrapWidth = 75
            q1Wrapped = "\n".join(textwrap.wrap(q1, width=wrapWidth))
            q2Wrapped = "\n".join(textwrap.wrap(q2, width=wrapWidth))
            labels.append(f"{q1Wrapped}\n        vs\n{q2Wrapped}")
            scores.append(score)

        y_pos = np.arange(len(labels)) * 2.5  # More spacing between bars

        # Create horizontal bar chart with gradient colors
        bars = ax.barh(
            y_pos,
            scores,
            color=colors,
            edgecolor="white",
            linewidth=2,
            height=1.8,
        )

        # Add score labels inside or outside bars depending on space
        for bar, score in zip(bars, scores):
            textColor = "white" if score > 0.5 else "#333333"
            xPos = bar.get_width() - 0.03 if score > 0.5 else bar.get_width() + 0.02
            ha = "right" if score > 0.5 else "left"

            ax.text(
                xPos,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}",
                va="center",
                ha=ha,
                fontsize=14,
                fontweight="bold",
                color=textColor,
            )

        # Styling
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9, linespacing=1.15, fontfamily="monospace")
        ax.invert_yaxis()
        ax.set_xlabel("Procrustes Similarity Score", fontsize=12, fontweight="medium")
        ax.set_xlim(0, 1.0)

        # Remove spines for cleaner look
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        # Add subtle grid
        ax.xaxis.grid(True, linestyle="--", alpha=0.3)
        ax.yaxis.grid(False)

        # Title
        ax.set_title(
            f"{indexName}\nTop 5 Question Pairs with Highest Correlation",
            fontsize=16,
            fontweight="bold",
            pad=20,
            color="#2d3748",
        )

        # Add tick styling
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", colors="#666666")

        plt.tight_layout()

        # Generate filename from index name
        safeFilename = indexName.replace(" ", "_") + "_top5_correlations.png"

        # Save the figure
        plt.savefig(
            os.path.join(dataFolder, safeFilename),
            bbox_inches="tight",
            dpi=150,
            facecolor="white",
            edgecolor="none",
        )
        plt.close()

        print(f"Graph saved: {safeFilename}")


def indexCorrelation() -> None:
    """
    This function calculates the correlation matrices for each index
    and generates visualizations including the top 5 correlated question pairs.

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
        print(f"\n{'='*80}")
        print(f"{g}")
        print(f"{'='*80}")
        print("Top 5 question pairs with highest correlation:")
        for i, (q1, q2, score) in enumerate(topPairs, 1):
            print(f"\n  {i}. Correlation: {score:.4f}")
            print(f"     Q1: {q1}")
            print(f"     Q2: {q2}")

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

    # Generate the combined visualization for top 5 correlations by index
    plotTopCorrelatedQuestions(allTopPairs, dataFolder)


if __name__ == "__main__":
    indexCorrelation()
