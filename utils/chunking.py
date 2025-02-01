import pandas as pd

def chunk_dataframe(df: pd.DataFrame, chunk_size: int = 200):
    """
    Yields successive chunks of the DataFrame, each chunk of size `chunk_size`.

    Args:
        df (pd.DataFrame): The DataFrame to be split.
        chunk_size (int): The max number of rows per chunk.

    Yields:
        pd.DataFrame: The next chunk of rows.
    """
    for start_idx in range(0, len(df), chunk_size):
        yield df.iloc[start_idx : start_idx + chunk_size]

def chunk_list(input_list:list, chunk_size:int) -> list:
    """
    Chunks a list into smaller lists of a defined chunk_size
    """
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]
