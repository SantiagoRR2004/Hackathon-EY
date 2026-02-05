import pandas as pd
import numpy as np


def loadCSV(filePath: str) -> pd.DataFrame:
    """
    This function loads a CSV file and returns a DataFrame.

    It also fixes any columns with vector data that are
    stored as strings in the CSV file.

    Args:
        - filePath: The path to the CSV file

    Returns:
        - A DataFrame containing the data from the CSV file
    """
    data = pd.read_csv(filePath)

    # Fix vectors
    for c in data.columns:
        if (
            pd.api.types.is_string_dtype(data[c])
            and data[c][0].startswith("[")
            and data[c][0].endswith("]")
        ):
            data[c] = data[c].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

    return data
