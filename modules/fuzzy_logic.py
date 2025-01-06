import pandas as pd
from thefuzz import fuzz

def python_pre_filter_fuzzy(
    df: pd.DataFrame,
    column_keywords: dict,
    threshold: int = 85
) -> (pd.DataFrame, list):
    """
    Exclude rows if the cell in ANY relevant column is a fuzzy match
    (ratio >= threshold) to ANY of the keywords.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        column_keywords (dict): A mapping of column_name -> list of exclude keywords.
        threshold (int): The fuzzy match threshold; default is 85 (0-100 range).
    
    Returns:
        (filtered_df, excluded_rows):
            - filtered_df: DataFrame of rows that remain after fuzzy matching.
            - excluded_rows: list of tuples describing excluded rows in the form:
                    (row_index, column, cell_value, matched_keyword, fuzzy_score)
    """
    keep_mask = [True] * len(df)
    excluded_rows = []

    for i, row in df.iterrows():
        # For each row, check each column's cell content
        for col, keywords in column_keywords.items():
            if col in df.columns:
                cell_value = str(row[col]).lower()
                for kw in keywords:
                    score = fuzz.ratio(cell_value, kw.lower())
                    if score >= threshold:
                        # Exclude this row
                        keep_mask[i] = False
                        excluded_rows.append(
                            (i, col, row[col], kw, score)
                        )
                        break  # Stop checking more keywords in this row

            # If this row is already excluded, no need to check further columns
            if not keep_mask[i]:
                break

    # Build a new DataFrame with only rows marked for 'keep'
    filtered_df = df[keep_mask].copy()
    return filtered_df, excluded_rows
