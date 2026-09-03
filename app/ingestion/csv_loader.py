import pandas as pd


def load_csv(file_path):
    """
    Extract data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        str: CSV data converted into text.
    """

    dataframe = pd.read_csv(file_path)

    text = dataframe.to_string(index=False)

    return text