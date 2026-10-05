import pandas as pd
import os


def load_csv(file_path: str) -> str:
    """
    Extract structured tabular data from a CSV file.
    Converts records into semantically rich representations suitable
    for dense vector embeddings and LLM reasoning.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        str: Cleaned, structured text representation of the CSV.
    """
    # Attempt common encodings
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    dataframe = None

    for enc in encodings:
        try:
            dataframe = pd.read_csv(file_path, encoding=enc)
            break
        except (UnicodeDecodeError, Exception):
            continue

    if dataframe is None:
        raise ValueError(f"Unable to read CSV file: {file_path}")

    # Remove completely empty rows and columns
    dataframe = dataframe.dropna(how="all")
    dataframe = dataframe.dropna(axis=1, how="all")

    if dataframe.empty:
        return "Empty CSV document."

    # Fill NaN with "N/A"
    dataframe = dataframe.fillna("N/A")

    # Format each record as key-value pairs for maximum RAG retrieval accuracy
    records = []
    headers = [str(col).strip() for col in dataframe.columns]

    for idx, row in dataframe.iterrows():
        items = []
        for h in headers:
            val = str(row[h]).strip()
            if val and val != "N/A":
                items.append(f"{h}: {val}")
        if items:
            records.append(" | ".join(items))

    # Also provide a clean markdown table header and rows
    table_header = " | ".join(headers)
    table_separator = " | ".join(["---"] * len(headers))
    table_rows = [
        " | ".join(str(row[h]).strip() for h in headers)
        for _, row in dataframe.iterrows()
    ]
    markdown_table = "\n".join([table_header, table_separator] + table_rows)

    filename = os.path.basename(file_path)
    output = f"DOCUMENT: {filename}\n\nSUMMARY TABLE:\n{markdown_table}\n\nSTRUCTURED RECORDS:\n" + "\n".join(
        f"- {rec}" for rec in records
    )

    return output