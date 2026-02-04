from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas
import math
import os


def binaryNoMissing(rawData: pandas.DataFrame, cleanedData: pandas.DataFrame) -> None:
    """
    This function is used to clean binary columns with no missing values.

    We simply move them to the cleaned data.

    Args:
        - rawData: The raw data
        - cleanedData: The cleaned data

    Returns:
        - None
    """
    # Find 0 and 1 columns with no missing values
    for column in rawData.columns:
        if not rawData[column].isnull().any() and set(
            rawData[column].dropna().unique()
        ).issubset({0, 1}):
            cleanedData[column] = rawData[column]
            del rawData[column]


def binaryMissing(rawData: pandas.DataFrame, cleanedData: pandas.DataFrame) -> None:
    """
    This function is used to clean binary columns with missing values.

    This is one hot encoding them considering missing values as a category.

    Args:
        - rawData: The raw data
        - cleanedData: The cleaned data

    Returns:
        - None
    """
    for column in rawData.columns:
        if set(rawData[column].dropna().unique()).issubset({0, 1}):
            cleanedData[column] = rawData[column].apply(
                lambda x: np.eye(3)[2 if pandas.isna(x) else int(x)]
            )
            del rawData[column]


def basicOneHot(rawData: pandas.DataFrame, cleanedData: pandas.DataFrame) -> None:
    """
    This function is used to clean columns that are categorical without order.
    If they have a missing value, it is another category.

    We look for:
        -  `Yes`/`No`
        -  `Yes`/`No`/`Maybe`
        -  `Yes`/`No`/`I don't know`
        -  `Yes`/`No`/`I'm not sure`
        -  `Yes`/`No`/`I am not sure`
        -  `Yes`/`No`/`Unsure`/`Not applicable to me`
        -  `Yes`/`No`/`I don't know`/`Not eligible for coverage / N/A`

    We one-hot encode them.

    Args:
        - rawData: The raw data
        - cleanedData: The cleaned data

    Returns:
        - None
    """
    validOneHot = [
        {"Yes", "No"},
        {"Yes", "No", "Maybe"},
        {"Yes", "No", "I don't know"},
        {"Yes", "No", "I'm not sure"},
        {"Yes", "No", "I am not sure"},
        {"Yes", "No", "Unsure", "Not applicable to me"},
        {"Yes", "No", "I don't know", "Not eligible for coverage / N/A"},
    ]

    for column in rawData.columns:
        hasMissing = int(rawData[column].isnull().any())
        uniqueValues = set(rawData[column].dropna().unique())

        if uniqueValues in validOneHot:
            categories = list(uniqueValues)  # Fixed order
            cleanedData[column] = rawData[column].apply(
                lambda x: np.eye(len(uniqueValues) + hasMissing)[
                    categories.index(x) if not pandas.isna(x) else -1
                ]
            )
            del rawData[column]


def companySize(rawData: pandas.DataFrame, cleanedData: pandas.DataFrame) -> None:
    """
    This function is used to clean the column:
        How many employees does your company or organization have?

    We map the ranges to their midpoints and apply a log transformation and
    Min-Max scaling. We also create a binary indicator for missing values.

    Args:
        - rawData: The raw data
        - cleanedData: The cleaned data

    Returns:
        - None
    """
    minMaxScaler = MinMaxScaler()
    sizeMapping = {
        "1-5": math.log((1 + 5) / 2),
        "6-25": math.log((6 + 25) / 2),
        "26-100": math.log((26 + 100) / 2),
        "100-500": math.log((100 + 500) / 2),
        "500-1000": math.log((500 + 1000) / 2),
        "More than 1000": math.log(1500),
        pandas.NA: -1,
    }
    scaled = minMaxScaler.fit_transform(
        rawData["How many employees does your company or organization have?"]
        .map(sizeMapping)
        .to_frame()
    ).ravel()
    cleanedData["How many employees does your company or organization have?"] = [
        np.array([v, int(notna)])
        for v, notna in zip(
            scaled,
            rawData[
                "How many employees does your company or organization have?"
            ].notna(),
        )
    ]
    del rawData["How many employees does your company or organization have?"]


def cleanData() -> None:
    """
    Clean the mental health dataset and save the cleaned and remaining data.

    Because we need to calculate the correlations between the initial columns, we can't
    create new columns (e.g., one-hot encoding). This means we if we use techniques
    that create new columns, we need to store them all in one vector.

    Args:
        - None

    Returns:
        - None
    """
    # Get the paths
    currentDirectory = os.path.dirname(os.path.abspath(__file__))
    dataFolder = os.path.join(currentDirectory, "data")

    # Load the raw data
    rawData = pandas.read_csv(os.path.join(dataFolder, "mental_health.csv"))

    # Print the number of rows and columns
    print(f"Number of rows: {rawData.shape[0]}")
    nColumns = rawData.shape[1]
    print(f"Number of columns: {nColumns}")

    cleanedData = pandas.DataFrame()

    # Find 0 and 1 columns with no missing values
    binaryNoMissing(rawData, cleanedData)

    # Find 0 and 1 columns with missing values
    binaryMissing(rawData, cleanedData)

    # Columns with only Yes/No/Maybe values
    basicOneHot(rawData, cleanedData)

    # How many employees does your company or organization have?
    companySize(rawData, cleanedData)

    # Print the number of columns
    print(f"Number of cleaned columns: {cleanedData.shape[1]}")
    print(f"Number of remaining columns: {rawData.shape[1]}")

    assert (
        cleanedData.shape[1] + rawData.shape[1] == nColumns
    ), "Some columns were lost during cleaning."

    # Save the cleaned data
    cleanedData.to_csv(
        os.path.join(dataFolder, "mental_healthCleaned.csv"), index=False
    )

    # Save the remaining raw data for further analysis
    rawData.to_csv(os.path.join(dataFolder, "mental_healthRemaining.csv"), index=False)


if __name__ == "__main__":
    cleanData()
