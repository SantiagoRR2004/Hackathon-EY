from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score
import pandas as pd
import correlation
import numpy as np
import utils
import os


def trainModel(X: pd.DataFrame, y: pd.Series) -> float:
    """
    This function trains a model using Leave-One-Out Cross Validation
    and returns the mean F1 score.

    Args:
        - X: The features to train the model on
        - y: The target variable to train the model on

    Returns:
        - float: The mean F1 score across all LOO iterations
    """
    # If the y is a numpy array (one hot encoded),
    # convert it back to a single column
    if isinstance(y.iloc[0], np.ndarray) and y.iloc[0].ndim == 1:
        y = y.apply(np.argmax)

    # Convert to numpy for sklearn compatibility
    XArray = X.values
    yArray = y.values

    # Leave-One-Out Cross Validation
    loo = LeaveOneOut()
    predictions = []
    trueValues = []

    for trainIndex, testIndex in loo.split(XArray):
        XTrain, XTest = XArray[trainIndex], XArray[testIndex]
        yTrain, yTest = yArray[trainIndex], yArray[testIndex]

        model = DecisionTreeClassifier(random_state=42)
        model.fit(XTrain, yTrain)

        yPred = model.predict(XTest)
        predictions.append(yPred[0])
        trueValues.append(yTest[0])

    # Calculate mean F1 score (macro for multiclass)
    predictions = np.array(predictions)
    trueValues = np.array(trueValues)

    # Determine if binary or multiclass
    uniqueClasses = np.unique(trueValues)
    if len(uniqueClasses) == 2:
        meanF1 = f1_score(
            trueValues, predictions, average="binary", pos_label=uniqueClasses[1]
        )
    else:
        meanF1 = f1_score(trueValues, predictions, average="macro")

    print(f"  Leave-One-Out CV Mean F1 Score: {meanF1:.4f}")
    print(f"  Accuracy: {(predictions == trueValues).sum() / len(trueValues):.4f}")

    return meanF1


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

    # Graph the entire correlation matrix
    utils.graphCorrelationMatrix(correlationMatrix, "All")

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

        utils.printSeparator(c)
        trainModel(correlations, cleanData[c])


if __name__ == "__main__":
    modeling()
