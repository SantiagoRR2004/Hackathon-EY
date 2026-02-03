import pandas
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

    # Find 0 and 1 columns with no missing values
    for column in rawData.columns:
        if not rawData[column].isnull().any() and set(
            rawData[column].dropna().unique()
        ).issubset({0, 1}):
            cleanedData[column] = rawData[column]
            del rawData[column]

    # Print the number of columns
    print(f"Number of cleaned columns: {cleanedData.shape[1]}")
    print(f"Number of remaining columns: {rawData.shape[1]}")

    # Save the cleaned data
    cleanedData.to_csv(
        os.path.join(dataFolder, "mental_healthCleaned.csv"), index=False
    )
