import pandas as pd
import correlation
import utils
import os


def trainModel(X: pd.DataFrame, y: pd.Series) -> None:
    """
    This function trains a model on the given data.

    Args:
        - X: The features to train the model on
        - y: The target variable to train the model on

    Returns:
        - None
    """
    # Train a model (e.g., logistic regression, random forest, etc.)
    pass


def modeling() -> None:
    """
    We get the 10 most correlated features with the target variable
    and train a model on those features.

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

    # Calculate the correlation matrix
    correlationMatrix = correlation.calculateCorrelationsMatrix(cleanData)

    columns = [
        "Do you currently have a mental health disorder?",
        "Have you ever sought treatment for a mental health issue from a mental health professional?",
    ]

    for c in columns:
        # Get the biggest 10 correlations with the target column
        correlations = correlationMatrix[c].sort_values(ascending=False)[:10]

        trainModel(cleanData[correlations.index], cleanData[c])


if __name__ == "__main__":
    modeling()
