import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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
