from openpyxl import Workbook
import json
import os

if __name__ == "__main__":
    currentDirectory = os.path.dirname(os.path.abspath(__file__))

    # Find all JSON files in the current directory
    jsonFiles = [f for f in os.listdir(currentDirectory) if f.endswith(".json")]

    for f in jsonFiles:
        with open(os.path.join(currentDirectory, f), "r") as file:
            data = json.load(file)

        # Create a new workbook and select the active worksheet
        workbook = Workbook()
        sheet = workbook.active

        # Header
        sheet.append(list(data["data"].keys()))

        # Row
        sheet.append(list(data["data"].values()))

        # Save the workbook with a name based on the original JSON file
        excelFileName = f.replace(".json", ".xlsx")
        workbook.save(os.path.join(currentDirectory, excelFileName))
