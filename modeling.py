from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import pandas as pd
import correlation
import numpy as np
import utils
import os


def trainModel(X: pd.DataFrame, y: pd.Series) -> float:
    """
    This function trains a model using Stratified K-Fold Cross Validation (5 splits)
    and calculates the mean F1 score.

    Args:
        - X (pd.DataFrame): The features to train the model on
        - y (pd.Series): The target variable to train the model on

    Returns:
        - float: The F1 score of the best model
    """
    # If the y is a numpy array (one hot encoded),
    # convert it back to a single column
    if isinstance(y.iloc[0], np.ndarray) and y.iloc[0].ndim == 1:
        y = y.apply(np.argmax)

    # Convert to numpy for sklearn compatibility
    XArray = X.values
    yArray = y.values

    uniqueClasses = np.unique(yArray)
    if len(uniqueClasses) == 2:
        f1Method = "binary"
    else:
        f1Method = "macro"

    models = {
        LogisticRegression(random_state=42): {"name": "Logistic Regression"},
        RandomForestClassifier(random_state=42): {"name": "Random Forest"},
        DecisionTreeClassifier(random_state=42): {"name": "Decision Tree"},
    }

    bestF1 = 0

    for model, modelInfo in models.items():

        f1Scores = []
        accuracies = []

        # Use StratifiedKFold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for trainIndex, testIndex in cv.split(XArray, yArray):
            XTrain, XTest = XArray[trainIndex], XArray[testIndex]
            yTrain, yTest = yArray[trainIndex], yArray[testIndex]

            model.fit(XTrain, yTrain)

            yPred = model.predict(XTest)

            f1Scores.append(f1_score(yTest, yPred, average=f1Method))
            accuracies.append((yPred == yTest).sum() / len(yTest))

        # Calculate mean F1 score (macro for multiclass)
        meanF1 = np.mean(f1Scores)
        meanAccuracy = np.mean(accuracies)

        print(f"{modelInfo['name']} (5) CV Mean F1 Score: {meanF1:.4f}")
        print(f"Accuracy: {meanAccuracy:.4f}")

        if meanF1 > bestF1:
            bestF1 = meanF1

    return bestF1


def modeling() -> dict:
    """
    We get the 10 most correlated features with the target variable
    and train a model on those features.

    Args:
        - None

    Returns:
        - dict: A dictionary containing the features and F1 scores for each target variable
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
    modelsData = {}

    for c in columns:
        # Get the biggest 10 correlations with the target column
        features = correlationMatrix[c].sort_values(ascending=False)[:10]

        # Show top ten features
        utils.plotTopCorrelatedQuestions(
            list(features.items()), dataFolder, title=c, versus=False
        )

        correlations = cleanData[features.index]

        # Divide columns that are vectors into their components
        for col in correlations.columns:
            correlations = correlations.join(
                pd.DataFrame(correlations.pop(col).tolist()).add_prefix(f"{col}_")
            )

        utils.printSeparator(c)
        f1Score = trainModel(correlations, cleanData[c])

        modelsData[c] = {
            "Features": features.tolist(),
            "F1Score": f1Score,
        }

    return modelsData


if __name__ == "__main__":
    modeling()
