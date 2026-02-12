import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import textwrap
import shutil
import os


def loadCSV(filePath: str) -> pd.DataFrame:
    """
    This function loads a CSV file and returns a DataFrame.

    It also fixes any columns with vector data that are
    stored as strings in the CSV file.

    Args:
        - filePath: The path to the CSV file

    Returns:
        - A DataFrame containing the data from the CSV file
    """
    data = pd.read_csv(filePath)

    # Fix vectors
    for c in data.columns:
        if (
            pd.api.types.is_string_dtype(data[c])
            and data[c][0].startswith("[")
            and data[c][0].endswith("]")
        ):
            data[c] = data[c].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

    return data


def graphCorrelationMatrix(matrix: pd.DataFrame, fileName: str) -> None:
    """
    This function graphs the correlation matrix as a heatmap.

    Args:
        - matrix: The correlation matrix to graph
        - fileName: The name of the file to save the graph to

    Returns:
        - None
    """
    # Get the paths
    currentDirectory = os.path.dirname(os.path.abspath(__file__))
    dataFolder = os.path.join(currentDirectory, "data")

    data = matrix.to_numpy()
    labels = matrix.columns

    fig, ax = plt.subplots(figsize=(len(labels), len(labels)))
    im = ax.imshow(data, aspect="equal", cmap="coolwarm")

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
    ax.grid(False)

    plt.title(fileName)
    plt.colorbar(im)

    # Save the figure
    plt.savefig(
        os.path.join(dataFolder, f"Correlation{fileName.replace(' ', '')}.png"),
        bbox_inches="tight",
    )
    plt.close()


def printSeparator(text: str, sep: str = "=") -> None:
    """
    Print a separator line with centered text.

    Args:
        - text (str): The text to center in the separator.
        - sep (str): The character to use for the separator line.

    Returns:
        - None
    """
    columns = shutil.get_terminal_size().columns

    inner = f" {text.strip()} "
    padding = (columns - len(inner)) // 2

    line = sep * padding + inner + sep * padding

    if len(line) < columns:
        line += sep

    print(sep * columns)
    print(line)
    print(sep * columns)


def plotTopCorrelatedQuestions(
    topPairs: list, dataFolder: str, title: str, versus: bool = True
) -> None:
    """
    Plot top correlated questions as horizontal bar charts.

    Args:
        - topPairs (list): dict mapping index names to lists of tuples.
            If versus is True, tuples are (q1, q2, score).
            If versus is False, tuples are (q, score).
        - dataFolder (str): Path to save the generated images.
        - title (str): Custom title for the chart.
        - versus (bool): If True, labels show "q1 vs q2". If False, labels show a single question.

    Returns:
        - None
    """
    plt.style.use("ggplot")

    labels = []
    scores = []

    for i, entry in enumerate(topPairs, 1):
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
    ax.grid(False)

    bars = ax.barh(labels, scores, color=plt.cm.coolwarm(scores), height=0.6)

    ax.bar_label(bars, fmt="%.3f", padding=8, fontweight="bold", fontsize=11)

    ax.tick_params(axis="y", labelsize=9)

    chartTitle = f"Top {len(topPairs)} Correlations: {title}"
    ax.set_title(chartTitle, fontsize=15, pad=20, fontweight="bold")
    ax.set_xlim(0, max(scores) * 1.18)
    ax.invert_yaxis()
    ax.set_xticks([])
    plt.tight_layout()

    # Generate filename from index name
    safeFilename = f"Top{len(topPairs)}{title.replace(' ', '')}.png"

    # Save the figure
    plt.savefig(
        os.path.join(dataFolder, safeFilename),
        bbox_inches="tight",
        dpi=150,
        facecolor="white",
        edgecolor="none",
    )
    plt.close()


def graphClusters(data: pd.DataFrame, labels: np.ndarray, title: str):
    """
    Graph clusters and save them to a png.

    Args:
        - data (pd.DataFrame): The data to graph
        - labels (np.ndarray): The cluster labels for each data point
        - title (str): The title of the graph

    Returns:
        - None
    """
    currentDirectory = os.path.dirname(os.path.abspath(__file__))
    dataFolder = os.path.join(currentDirectory, "data")
    plt.figure()

    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap="viridis")
    plt.title(f"{title} Clusters")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

    plt.savefig(
        os.path.join(dataFolder, f"Clusters{title.replace(' ', '')}.png"),
        bbox_inches="tight",
    )
    plt.close()
