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
        -  `Yes`/`No`/`I'm not sure`/`Not applicable to me`
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
        {"Yes", "No", "I'm not sure", "Not applicable to me"},
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
    mapped = rawData["How many employees does your company or organization have?"].map(
        sizeMapping
    )
    scaled = minMaxScaler.fit_transform(mapped[mapped != -1].to_frame()).ravel()
    scaledFull = pandas.Series(-1, index=mapped.index, dtype=float)
    scaledFull[mapped != -1] = scaled.ravel()
    cleanedData["How many employees does your company or organization have?"] = [
        np.array([v, int(notna)])
        for v, notna in zip(
            scaledFull,
            rawData[
                "How many employees does your company or organization have?"
            ].notna(),
        )
    ]
    del rawData["How many employees does your company or organization have?"]


def basicOrdinal(rawData: pandas.DataFrame, cleanedData: pandas.DataFrame) -> None:
    """
    This function is used to clean columns that have values related to an ordinal scale
    and maybe other that are completely independent.

    For each independent category, we create a binary indicator and use -1 in the
    ordinal column. For the ordinal values, we map them to a scale from 0 to 1.

    If there are missing values, we treat them the same way as the independent categories.

    Args:
        - rawData: The raw data
        - cleanedData: The cleaned data

    Returns:
        - None
    """
    mappings = [
        {
            "Very easy": 0,
            "Somewhat easy": 0.25,
            "Neither easy nor difficult": 0.5,
            "Somewhat difficult": 0.75,
            "Very difficult": 1,
            "I don't know": -1,
        },
        {
            "No, I don't know any": 0,
            "I know some": 0.5,
            "Yes, I know several": 1,
        },
        {
            "No, because it would impact me negatively": 0,
            "Sometimes, if it comes up": 0.5,
            "Yes, always": 1,
            "No, because it doesn't matter": -1,  # TODO
            "Not applicable to me": -1,
        },
        {
            "No, none did": 0,
            "Some did": 0.5,
            "Yes, they all did": 1,
            "I don't know": -1,
        },
        {
            "None did": 0,
            "Some did": 0.5,
            "Yes, they all did": 1,
            "I don't know": -1,
        },
        {
            "N/A (not currently aware)": 0,
            "I was aware of some": 0.5,
            "Yes, I was aware of all of them": 1,
            "No, I only became aware later": -1,
        },
        {
            "No": 0,
            "Sometimes": 0.5,
            "Yes, always": 1,
            "I don't know": -1,
        },
        {
            "None of them": 0,
            "Some of them": 0.5,
            "Yes, all of them": 1,
        },
        {
            "None of them": 0,
            "Some of them": 0.5,
            "Yes, all of them": 1,
            "I don't know": -1,
        },
        {
            "None did": 0,
            "Some did": 0.5,
            "Yes, they all did": 1,
        },
        {
            "No, at none of my previous employers": 0,
            "Some of my previous employers": 0.5,
            "Yes, at all of my previous employers": 1,
        },
        {
            "No, at none of my previous employers": 0,
            "Some of my previous employers": 0.5,
            "Yes, at all of my previous employers": 1,
            "I don't know": -1,
        },
        {
            "Not open at all": 0,
            "Somewhat not open": 0.25,
            "Neutral": 0.5,
            "Somewhat open": 0.75,
            "Very open": 1,
            "Not applicable to me (I do not have a mental illness)": -1,
        },
        {
            "Never": 0,
            "Rarely": 1 / 3,
            "Sometimes": 2 / 3,
            "Often": 1,
            "Not applicable to me": -1,
        },
        {
            "Never": 0,
            "Sometimes": 0.5,
            "Always": 1,
        },
        {
            "1-25%": 0,
            "26-50%": 1 / 3,
            "51-75%": 2 / 3,
            "76-100%": 1,
        },
    ]

    for column in rawData.columns:
        hasMissing = int(rawData[column].isnull().any())
        uniqueValues = set(rawData[column].dropna().unique())

        for mapping in mappings:
            if uniqueValues == set(mapping.keys()):

                # Basic Mapping
                values = [mapping.get(x, -1) for x in rawData[column]]

                # -1 values
                for key, value in mapping.items():
                    if value == -1:
                        values = [
                            (v if isinstance(v, list) else [v]) + [int(x == key)]
                            for v, x in zip(values, rawData[column])
                        ]

                # Missing value column
                if hasMissing:
                    values = [
                        (v if isinstance(v, list) else [v]) + [int(pandas.isna(x))]
                        for v, x in zip(values, rawData[column])
                    ]

                cleanedData[column] = [np.array(v) for v in values]
                del rawData[column]


def pipes(rawData: pandas.DataFrame, cleanedData: pandas.DataFrame) -> None:
    """
    This function is used to clean columns that use pipes to separate multiple values.

    If the category is present, we set the corresponding index to 1, otherwise 0.

    Args:
        - rawData: The raw data
        - cleanedData: The cleaned data

    Returns:
        - None
    """
    columnNames = [
        "If maybe, what condition(s) do you believe you have?",
        "If so, what condition(s) were you diagnosed with?",
        "Which of the following best describes your work position?",
    ]

    for column in columnNames:
        uniqueValues = set()
        for entry in rawData[column].dropna().unique():
            for value in entry.split("|"):
                uniqueValues.add(value.strip())

        uniqueValues = list(uniqueValues)  # Fixed order

        cleanedData[column] = rawData[column].apply(
            lambda x: np.eye(len(uniqueValues))[
                (
                    [uniqueValues.index(value.strip()) for value in x.split("|")]
                    if not pandas.isna(x)
                    else []
                )
            ].sum(axis=0)
        )
        del rawData[column]


def gender(rawData: pandas.DataFrame, cleanedData: pandas.DataFrame) -> None:
    """
    This function is used to clean the gender column.
        What is your gender?

    TODO Expand the other category to divide the different types of
    gender that are not male or female.

    Args:
        - rawData (pd.DataFrame): The raw data
        - cleanedData (pd.DataFrame): The cleaned data

    Returns:
        - None
    """
    male = {
        "male",
        "m",
        "man",
        "m|",
        "malr",
        "male.",
        "mail",
        "cis man",
        "dude",
        "cisdude",
        "male (cis)",
        "cis male",
        "sex is male",
        "i'm a man why didn't you make this a drop down question. you should of asked sex? and i would of answered yes please. seriously how much text can this take?",
    }
    female = {
        "female",
        "f",
        "fem",
        "fm",
        "female/woman",
        "woman",
        "cis-woman",
        "cis female",
        "cisgender female",
        "female assigned at birth",
        "i identify as female.",
        "female-bodied; no feelings about gender",
        "female (props for making this a freeform field, though)",
    }

    # # Uncomment to see the other values
    # uniqueValues = set(
    #     x.strip().lower() for x in rawData["What is your gender?"].dropna().unique()
    # )
    # print(uniqueValues.difference(set.union(male, female)))

    binary = set.union(male, female)

    cleanedData["What is your gender?"] = rawData["What is your gender?"].apply(
        lambda x: np.array(
            [
                int(x.strip().lower() in male) if pandas.notna(x) else 0,
                int(x.strip().lower() in female) if pandas.notna(x) else 0,
                (int(x.strip().lower() not in binary) if pandas.notna(x) else 1),
            ]
        )
    )
    del rawData["What is your gender?"]


def stringColumns(rawData: pandas.DataFrame, cleanedData: pandas.DataFrame) -> None:
    """
    This function is used to clean columns that are free-form text.

    TODO For now we just use the length of the text.

    Args:
        - rawData: The raw data
        - cleanedData: The cleaned data

    Returns:
        - None
    """
    stringColumns = ["Why or why not?", "Why or why not?.1"]

    for column in stringColumns:
        cleanedData[column] = rawData[column].apply(
            lambda x: len(x) if pandas.notna(x) else 0
        )
        del rawData[column]


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

    # Columns with pipe-separated values
    pipes(rawData, cleanedData)

    # Columns with basic ordinal values
    basicOrdinal(rawData, cleanedData)

    # Gender column
    gender(rawData, cleanedData)

    # Free-form text columns
    stringColumns(rawData, cleanedData)

    # Print the number of columns
    print(f"Number of cleaned columns: {cleanedData.shape[1]}")
    print(f"Number of remaining columns: {rawData.shape[1]}")

    assert (
        cleanedData.shape[1] + rawData.shape[1] == nColumns
    ), "Some columns were lost during cleaning."

    # Load the original data
    ogData = pandas.read_csv(os.path.join(dataFolder, "mental_health.csv"))

    # Reorder the cleaned and raw data to match the original data
    cleanedData = cleanedData[ogData.columns.intersection(cleanedData.columns)]
    rawData = rawData[ogData.columns.intersection(rawData.columns)]

    # Save the cleaned data
    cleanedData.to_csv(
        os.path.join(dataFolder, "mental_healthCleaned.csv"), index=False
    )

    # Save the remaining raw data for further analysis
    rawData.to_csv(os.path.join(dataFolder, "mental_healthRemaining.csv"), index=False)


if __name__ == "__main__":
    cleanData()
