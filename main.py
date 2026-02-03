from sklearn.preprocessing import MinMaxScaler
import pandas
import math
import os

if __name__ == "__main__":
    # Get the paths
    currentDirectory = os.path.dirname(os.path.abspath(__file__))
    dataFolder = os.path.join(currentDirectory, "data")

    # Load the raw data
    rawData = pandas.read_csv(os.path.join(dataFolder, "mental_health.csv"))

    # Print the number of rows and columns
    print(f"Number of rows: {rawData.shape[0]}")
    print(f"Number of columns: {rawData.shape[1]}")

    cleanedData = pandas.DataFrame()

    minMaxScaler = MinMaxScaler()

    # Find 0 and 1 columns with no missing values
    for column in rawData.columns:
        if not rawData[column].isnull().any() and set(
            rawData[column].dropna().unique()
        ).issubset({0, 1}):
            cleanedData[column] = rawData[column]
            del rawData[column]

    # Columns with only Yes/No/Maybe values
    for column in rawData.columns:
        if not rawData[column].isnull().any() and set(
            rawData[column].dropna().unique()
        ).issubset({"Yes", "No", "Maybe"}):
            oneHot = pandas.get_dummies(rawData[column], prefix=column)
            cleanedData = pandas.concat([cleanedData, oneHot], axis=1)
            del rawData[column]

    # How many employees does your company or organization have?
    sizeMapping = {
        "1-5": math.log((1 + 5) / 2),
        "6-25": math.log((6 + 25) / 2),
        "26-100": math.log((26 + 100) / 2),
        "100-500": math.log((100 + 500) / 2),
        "500-1000": math.log((500 + 1000) / 2),
        "More than 1000": math.log(1500),
        pandas.NA: -1,
    }
    cleanedData["How many employees does your company or organization have?"] = (
        minMaxScaler.fit_transform(
            rawData["How many employees does your company or organization have?"]
            .map(sizeMapping)
            .to_frame()
        )
    )
    cleanedData["CompanySizeKnown"] = (
        rawData["How many employees does your company or organization have?"].notna()
    ).astype(int)
    del rawData["How many employees does your company or organization have?"]

    # Print the number of columns
    print(f"Number of cleaned columns: {cleanedData.shape[1]}")
    print(f"Number of remaining columns: {rawData.shape[1]}")

    # Save the cleaned data
    cleanedData.to_csv(
        os.path.join(dataFolder, "mental_healthCleaned.csv"), index=False
    )

    # Save the remaining raw data for further analysis
    rawData.to_csv(os.path.join(dataFolder, "mental_healthRemaining.csv"), index=False)
