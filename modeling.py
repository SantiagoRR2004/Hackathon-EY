from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import correlation
import numpy as np
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
    # If the y is a numpy, it is one hot encoded,
    # so we need to convert it back to a single column
    if isinstance(y.iloc[0], np.ndarray) and y.iloc[0].ndim == 1:
        y = y.apply(np.argmax)

    # Divide 80-20 train-test split
    splitIndex = int(0.8 * len(X))
    XTrain, XTest = X[:splitIndex], X[splitIndex:]
    yTrain, yTest = y[:splitIndex], y[splitIndex:]

    model = DecisionTreeClassifier()
    model.fit(XTrain, yTrain)

    # Calculate F1 score
    yPred = model.predict(XTest)
    f1Score = (2 * (yTest == yPred).sum()) / (len(yTest) + len(yPred))
    print(f"F1 Score: {f1Score}")


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
        correlations = cleanData[
            correlationMatrix[c].sort_values(ascending=False)[:10].index
        ]

        # Divide columns that are vectors into their components
        for col in correlations.columns:
            correlations = correlations.join(
                pd.DataFrame(correlations.pop(col).tolist()).add_prefix(f"{col}_")
            )

        print(c)
        trainModel(correlations, cleanData[c])


if __name__ == "__main__":
    modeling()
