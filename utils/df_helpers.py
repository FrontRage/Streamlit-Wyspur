import pandas as pd

def summarize_row(row: pd.Series, index: int, columns_to_summarize: list) -> str:
    """
    Produce a concise summary string for one row, including the row index.
    
    Args:
        row (pd.Series): A row of data from the DataFrame.
        index (int): The row index in the DataFrame.
        columns_to_summarize (list): The columns we want to include in the summary.

    Returns:
        str: A textual summary of the row's content, referencing the row index.
    """
    col_summaries = []
    for col in columns_to_summarize:
        # Safely get the value, default to "N/A" if the column doesn't exist
        value = row.get(col, "N/A")
        col_summaries.append(f"{col}: {value}")
    
    summary_string = f"RowIndex: {index}, " + "; ".join(col_summaries)
    return summary_string


def make_chunk_summaries(chunk_df: pd.DataFrame, columns_to_summarize: list) -> list:
    """
    Build a list of row summaries for the given chunk of the DataFrame.
    
    Args:
        chunk_df (pd.DataFrame): The chunk of rows to summarize.
        columns_to_summarize (list): List of columns to include in the summary.
    
    Returns:
        list: A list of summary strings, one per row.
    """
    row_summaries = []
    for i, row in chunk_df.iterrows():
        row_summaries.append(summarize_row(row, i, columns_to_summarize))
    return row_summaries
