import json
import csv
import os

if __name__ == "__main__":
    currentDirectory = os.path.dirname(os.path.abspath(__file__))

    # Find all JSON files in the current directory
    jsonFiles = [f for f in os.listdir(currentDirectory) if f.endswith(".json")]

    for f in jsonFiles:
        with open(os.path.join(currentDirectory, f), "r") as file:
            data = json.load(file)

        csvFileName = f.replace(".json", ".csv")
        with open(
            os.path.join(currentDirectory, csvFileName),
            "w",
            newline="",
            encoding="utf-8",
        ) as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            writer.writerow(data["data"].keys())
            writer.writerow(data["data"].values())
