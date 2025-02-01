import streamlit as st
import pandas as pd
import sys
from io import StringIO
import json
import re
from time import sleep
from thefuzz import fuzz
from collections import defaultdict
# We'll import from our utils modules:
from utils.chunking import chunk_dataframe
from utils.df_helpers import make_chunk_summaries
from utils.prompt_builders import build_llm_prompt_single_col_json
from utils.chunking import chunk_list

# We'll import our OpenAI function from openai_module
from modules.openai_module import generate_text_basic
from modules.openai_module import generate_text_with_function_call

#### VISUAL MAP OF FUCNTION CALLING###

#filter_df_master
#├── filter_df_via_llm_per_column
#│   ├── chunk_dataframe (from utils.chunking)
#│   ├── build_llm_prompt_single_col_json (from utils.prompt_builders)
#│   ├── generate_text_basic (from modules.openai_module)
#│   └── parse_llm_decisions_single_col_json
#│       └── clean_and_parse_json # no longer used because we expect funtion calling form openai to return JSON
#├── aggregate_column_decisions
#└── filter_df_by_aggregated_decisions

def clean_and_parse_json(llm_response: str) -> list:
    """
    Cleans and parses JSON from the LLM response. Handles issues like unescaped backslashes,
    incomplete JSON, and ensures robust parsing.

    Args:
        llm_response (str): The raw JSON string from the LLM.

    Returns:
        list: Parsed JSON as Python objects (list of dictionaries), or None if parsing fails.
    """

    try:
        # Step 1: Fix unescaped backslashes
        cleaned_response = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', llm_response)

        # Step 2: Ensure the response starts and ends correctly
        if not cleaned_response.strip().startswith("["):
            cleaned_response = "[" + cleaned_response.lstrip()
        if not cleaned_response.strip().endswith("]"):
            cleaned_response = cleaned_response.rstrip() + "]"

        # Step 3: Attempt to parse the cleaned JSON
        parsed_data = json.loads(cleaned_response)

        # Step 4: Validate the parsed data is a list of dictionaries
        if not isinstance(parsed_data, list) or not all(isinstance(item, dict) for item in parsed_data):
            raise ValueError("Parsed JSON is not a list of dictionaries.")

        return parsed_data

    except (json.JSONDecodeError, ValueError) as e:
        # Debugging information for troubleshooting
        print(f"Error parsing JSON: {e}")
        print(f"Original LLM response: {llm_response}")
        print(f"Cleaned LLM response: {cleaned_response}")
        return None


def parse_llm_decisions_single_col_json(
    llm_response: list,
    valid_indices: set,
    debug: bool = False,
) -> dict:
    """
    Process a structured JSON response from the LLM for a single column chunk.

    Parameters
    ----------
    llm_response : list
        A structured JSON response from the LLM, where each item is a dictionary 
        containing "RowIndex" and "Decision".
    valid_indices : set
        Set of valid row indices for this chunk.
    debug : bool, optional
        Whether to include debug information, by default False.

    Returns
    -------
    dict
        A dictionary mapping row indices to decisions ("KEEP" or "EXCLUDE").
    """
    decisions = {}

    try:
        for item in llm_response:
            row_idx = item.get("RowIndex")
            decision_str = item.get("Decision", "").strip().upper()

            # Validate row index and decision
            if isinstance(row_idx, int) and row_idx in valid_indices:
                if decision_str in ["KEEP", "EXCLUDE"]:
                    decisions[row_idx] = decision_str
                else:
                    if debug:
                        st.warning(f"Invalid decision value skipped: {decision_str} for RowIndex: {row_idx}")
            else:
                if debug:
                    st.warning(f"Invalid row or index out of bounds: {item}")

    except Exception as e:
        st.error(f"Error processing LLM response: {e}")
        if debug:
            st.json(llm_response)

    return decisions




# def filter_df_via_llm_per_column(
#     df: pd.DataFrame,
#     columns_to_check: list,
#     filter_column_config: dict,
#     chunk_size: int,
#     reasoning_text: str,
#     model: str,
#     temperature: float,
#     debug: bool = False
# ) -> dict:
#     """
#     Step 2 (JSON-Based Version): For each column, we do a separate LLM call per chunk.
#     The LLM should return STRICT JSON output (one array of objects per prompt).
#     Return decisions_per_column = {
#       colName: { row_idx: "KEEP"/"EXCLUDE", ... },
#       ...
#     }

#     Parameters
#     ----------
#     df : pd.DataFrame
#         The full DataFrame to filter.
#     columns_to_check : list of str
#         Column names the user wants to filter on.
#     filter_column_config : dict
#         A dict containing filter config for each column, including logic and keywords.
#     chunk_size : int
#         Number of rows per chunk to send in one prompt.
#     reasoning_text : str
#         Conceptual instructions (e.g. from the slider).
#     model : str
#         Which LLM model to call (e.g., "gpt-3.5-turbo").
#     temperature : float
#         Sampling temperature for the LLM call.
#     debug : bool, optional
#         If True, we show prompt/response expansions and parse details.

#     Returns
#     -------
#     dict
#         A nested dict mapping each column -> {row_idx -> "KEEP" or "EXCLUDE"}.

#     Notes
#     -----
#     - Replaces the old pipe-delimited approach with a JSON approach for more reliable parsing.
#     - Uses build_llm_prompt_single_col_json to build the prompt and
#       parse_llm_decisions_single_col_json to parse the returned JSON.
#     - For row-level decisions, see aggregate_column_decisions() and filter_df_by_aggregated_decisions().
#     """
#     import logging
#     from thefuzz import fuzz # <---- IMPORT FUZZ HERE - IMPORTANT!

#     logging.basicConfig(
#         filename="llm_debug.log",
#         level=logging.DEBUG,
#         format="%(asctime)s - %(levelname)s - %(message)s"
#     )

#     decisions_per_column = {}
#     if df.empty:
#         return decisions_per_column

#     total_rows = len(df)
#     total_chunks = (total_rows // chunk_size) + (1 if total_rows % chunk_size else 0)

#     progress_bar = st.progress(0)
#     chunks_done = 0

#     # Optional debug container to avoid clutter
#     debug_container = st.container() if debug else None # <----- Fixed typo: st.container.container() -> st.container()

#     # We'll chunk the DataFrame
#     for chunk_idx, chunk_df in enumerate(chunk_dataframe(df, chunk_size), start=1):
#         valid_indices = set(chunk_df.index)

#         # Show progress in debug
#         if debug_container:
#             with debug_container:
#                 st.markdown(f"### Debug: Chunk {chunk_idx}/{total_chunks}")

#         # For each column the user wants to check
#         for col in columns_to_check:
#             # --- Get column-specific config from filter_column_config ---
#             col_config = filter_column_config.get(col, {}) # Get config for this column
#             keywords_str = col_config.get('keywords', "") # Get keywords as a STRING
#             logic = col_config.get('logic', "Conceptual Reasoning") # Get logic, default to Conceptual Reasoning
#             user_context = col_config.get('user_context', "") # Get user context (if any)

#             # --- SPLIT KEYWORD STRING INTO A LIST HERE ---
#             keywords_list = [kw.strip() for kw in keywords_str.split(',')] # Split by comma, strip whitespace
#             # Now 'keywords_list' is a list of strings (e.g., ['CEO', 'VP of Product', 'Director'])

#             st.write(f"--- Debug: Column: {col}, Logic: {logic}") # <--- DEBUG PRINT: LOGIC
#             st.write(f"--- Debug: Column: {col}, Keywords: {keywords_list}") # <--- DEBUG PRINT: KEYWORDS
#             st.write(f"--- Debug: Column: {col}, User Context: {user_context}") # <--- DEBUG PRINT: USER CONTEXT

#             if not keywords_list: # Use the LIST of keywords
#                 continue  # No keywords => skip this column


#             # --- CONDITIONAL LOGIC BASED ON 'logic' SETTING ---
#             if logic == "Conceptual Reasoning":
#                 st.write(f"--- Debug: Column: {col}, Applying Conceptual Reasoning Logic (LLM will be called)") # <--- Placeholder message
#                 # ... (Conceptual Reasoning logic will go here in future steps) ...
#                 # For now, just proceed with the existing prompt building and LLM call (for Conceptual Reasoning)

#             elif logic == "Exact Match":
#                 st.write(f"--- Debug: Column: {col}, Applying Exact Match Logic") # Debug message
#                 col_decisions = {} # Initialize decisions for this column

#                 for row_idx, row_data in chunk_df.iterrows():
#                     cell_value = str(row_data[col]).lower()
#                     is_match = False # Assume no match initially

#                     for keyword in keywords_list:
#                         if cell_value == keyword.lower().strip(): # Exact match (case-insensitive, whitespace stripped)
#                             is_match = True
#                             break # Exit keyword loop if exact match found
#                         elif fuzz.ratio(cell_value, keyword.lower().strip()) >= 85: # Fuzzy match (threshold 85)
#                             is_match = True
#                             break # Exit keyword loop if fuzzy match found

#                     # --- REVERSED KEEP/EXCLUDE LOGIC HERE ---
#                     if is_match:
#                         col_decisions[row_idx] = "KEEP" # <--- CORRECTED: KEEP if exact or fuzzy match
#                     else:
#                         col_decisions[row_idx] = "EXCLUDE" # <--- CORRECTED: EXCLUDE if NOT exact or fuzzy match

#                 decisions_per_column[col] = col_decisions # Store decisions for this column
#                 continue # <--- IMPORTANT: Skip LLM call for "Exact Match" - CONTINUE TO NEXT COLUMN

#             elif logic == "User Context":
#                 st.write(f"--- Debug: Column: {col}, Applying User Context Logic (Placeholder - LLM will be called)") # Debug message
#                 # ... (User Context logic will go here in Phase 4.4) ...
#                 # For now, fall through to Conceptual Reasoning (LLM call below)

#             else: # Conceptual Reasoning (and default case)
#                 st.warning(f"Unknown logic type '{logic}' for column '{col}'. Defaulting to Conceptual Reasoning.")
#                 st.write(f"--- Debug: Column: {col}, Applying Conceptual Reasoning Logic (Default - LLM will be called)") # <--- Placeholder message
#                 # ... (Conceptual Reasoning logic will go here in future steps) ...
#                 # Fall through to prompt building and LLM call below


#             # 1) Build row summaries for JUST this column
#             row_summaries = []
#             for row_idx, row_data in chunk_df.iterrows():
#                 text_for_llm = str(row_data[col])
#                 summary = f"RowIndex={row_idx}|Column='{col}'|Text='{text_for_llm}'"
#                 row_summaries.append(summary)

#             # 2) Build a JSON-based prompt
#             prompt = build_llm_prompt_single_col_json(
#                 row_summaries=row_summaries,
#                 column_name=col,
#                 keywords=keywords_list, # <--- Pass the LIST of keywords here
#                 reasoning_text=reasoning_text,
#                 user_context=user_context # <--- NEW: Pass user_context here!
#             )

#             # Debug: Show the prompt
#             if debug_container:
#                 with debug_container.expander(f"Prompt for Column '{col}', Chunk {chunk_idx} (JSON)", expanded=False):
#                     st.code(prompt, language="markdown")

#             # Generate the LLM response
#             try:
#                 llm_response = generate_text_with_function_call(
#                     prompt=prompt,
#                     model=model,
#                     temperature=temperature,
#                     debug=debug
#                 )

#                 # Debug: Show the structured JSON response
#                 if debug_container:
#                     with debug_container.expander(f"LLM JSON Response for Column '{col}', Chunk {chunk_idx} (Structured JSON)", expanded=False):
#                         st.json(llm_response)

#                 # Parse JSON decisions
#                 col_decisions = parse_llm_decisions_single_col_json(
#                     llm_response=llm_response,
#                     valid_indices=valid_indices,
#                     debug=debug
#                 )

#             except Exception as e:
#                 error_message = f"Failed to parse LLM response for Column '{col}', Chunk {chunk_idx}: {e}"
#                 st.error(error_message)
#                 logging.error(error_message)

#                 st.json(llm_response)  # Always show the raw JSON response
#                 continue

#             # Debug: Show parsed decisions
#             if debug_container:
#                 with debug_container.expander(f"Parsed JSON Decisions for Column '{col}', Chunk {chunk_idx}", expanded=False):
#                     st.write(col_decisions)

#             # Update decisions_per_column
#             if col not in decisions_per_column:
#                 decisions_per_column[col] = {}
#             decisions_per_column[col].update(col_decisions)

#         # Update progress bar
#         chunks_done += 1
#         progress_bar.progress(chunks_done / total_chunks)

#     return decisions_per_column





def aggregate_column_decisions(decisions_per_column: dict, debug: bool = False) -> (dict, dict):
    """
    Given a dictionary of the form:
        {
          "ColumnA": {0: "KEEP", 1: "EXCLUDE", ...},
          "ColumnB": {0: "KEEP", 1: "KEEP",    ...},
          ...
        }
    Return two dictionaries:
        1) final_decisions[row_index] = "KEEP" or "EXCLUDE"
        2) exclusion_reasons[row_index] = list of columns that triggered "EXCLUDE"

    Logic: a row_index is "KEEP" only if *all* columns say "KEEP" for that row.
           If a row doesn't appear in a particular column's dictionary,
           or if that column says "EXCLUDE," the row becomes "EXCLUDE."
    """
    from collections import defaultdict
    import streamlit as st

    final_decisions = {}
    # We'll accumulate all the columns that caused each row_idx to be "EXCLUDE"
    exclusion_reasons = defaultdict(list)  # row_idx -> list of columns that said EXCLUDE

    # Gather all row indices that appear in *any* column's decision dict
    all_row_indices = set()
    for col_name, decision_map in decisions_per_column.items():
        all_row_indices.update(decision_map.keys())

    # For each row that appears anywhere, decide if it's overall KEEP or EXCLUDE
    for row_idx in all_row_indices:
        keep_for_all_columns = True

        for col_name, decision_map in decisions_per_column.items():
            # If row_idx is missing from decision_map => treat that as EXCLUDE
            decision_for_this_col = decision_map.get(row_idx, "EXCLUDE")

            if decision_for_this_col == "EXCLUDE":
                keep_for_all_columns = False
                # Record which column(s) triggered EXCLUDE
                exclusion_reasons[row_idx].append(col_name)
                # Since we only need to know that at least one column excludes it,
                # we can break early. (But you can remove this break if you want
                # to list *all* columns that say EXCLUDE).
                # break

        # Final outcome for this row
        if keep_for_all_columns:
            final_decisions[row_idx] = "KEEP"
        else:
            final_decisions[row_idx] = "EXCLUDE"

    # Debug info if needed
    if debug:
        with st.expander("Final Aggregated Row Decisions (Debug)", expanded=False):
            st.write(final_decisions)
        with st.expander("Exclusion Reasons (Columns Triggering EXCLUDE)", expanded=False):
            # Convert the defaultdict to a normal dict for cleaner display
            st.write(dict(exclusion_reasons))

    # Return both (final decisions + which columns triggered exclude)
    return final_decisions, dict(exclusion_reasons)


def filter_df_by_aggregated_decisions(
    df: pd.DataFrame,
    final_decisions: dict
) -> pd.DataFrame:
    """
    Given the row-level decisions (with row indexes as keys),
    return a new DataFrame containing only rows labeled 'KEEP'.
    """
    keep_indices = [idx for idx, dec in final_decisions.items() if dec == "KEEP"]
    filtered_df = df.loc[keep_indices]
    return filtered_df





#######################################
# STEP 2: Replace filter_df_master
#######################################

def filter_df_master(
    df: pd.DataFrame,
    columns_to_check: list,
    filter_column_config: dict, 
    chunk_size: int,
    model: str,
    temperature: float,
    reasoning_text: str,
    debug: bool = False
) -> (pd.DataFrame, pd.DataFrame):
    """
    Filters the DataFrame across multiple columns (AND logic).
    Returns two DataFrames:
      1) filtered_df (KEEP rows),
      2) excluded_df (EXCLUDE rows).
    
    Key idea:
      - We do NOT remove excluded rows mid-process.
      - We track final_decisions[row] in a global dict.
      - Any row that fails a column is marked EXCLUDE + given reasons.
      - After all columns are processed, we split into KEPT and EXCLUDED sets.
    """

    # A. Initialize final decisions and reasons for ALL rows
    final_decisions = { idx: "KEEP" for idx in df.index }  # default keep
    decision_reasons = defaultdict(list)

    # B. Loop over columns in order
    for col in columns_to_check:
        if debug:
            print(f"\n=== Filter pass for column '{col}' ===")

        # 1) Identify the subset of rows STILL marked as "KEEP"
        keep_rows = [i for i, dec in final_decisions.items() if dec == "KEEP"]
        if not keep_rows:
            # if no rows left are keep, no reason to continue
            if debug:
                print(f"All rows already excluded before column '{col}'.")
            break

        # 2) Gather config & keywords for this column
        col_config = filter_column_config.get(col, {})
        keywords_str = col_config.get("keywords", "")
        user_keywords = set(kw.strip() for kw in keywords_str.split(",") if kw.strip())

        # 3) Extract unique texts *only from keep_rows*
        df_keep = df.loc[keep_rows]  # sub-DataFrame with keep rows
        unique_values_list = get_unique_column_values(df_keep, [col], debug)

        # 4) EXACT MATCH auto keep (technically they are already "KEEP",
        #    but let's add a reason if it's an exact match).
        to_send_to_llm = []
        for item in unique_values_list:
            text_value = item["Text"]
            if text_value in user_keywords:
                # All rows with this text are definitely keep
                # They are *already* keep, so just add reason:
                for idx in item["AllIndexes"]:
                    decision_reasons[idx].append(
                        f"Auto-KEEP (exact match) for column '{col}'"
                    )
            else:
                to_send_to_llm.append(item)

        # 5) LLM-based filtering for the rest
        chunked_unique_values = chunk_list(to_send_to_llm, chunk_size)
        for chunk in chunked_unique_values:
            col_keywords_list = list(user_keywords)  # or your logic
            prompt = build_llm_prompt_single_col_json(
                row_summaries=[
                    f'{{"RowIndex": {item["RowIndex"]}, "Text": "{item["Text"]}"}}'
                    for item in chunk
                ],
                column_name=col,
                keywords=col_keywords_list,
                reasoning_text=reasoning_text,
                user_context=col_config.get("user_context", "")
            )

            if debug:
                print(f"\nPrompt for column '{col}':\n{prompt}\n")

            llm_output = generate_text_with_function_call(
                prompt=prompt,
                model=model,
                temperature=temperature,
                debug=debug
            )

            if debug:
                print(f"\nLLM output for column '{col}':\n{llm_output}\n")

            llm_decisions = parse_llm_output(json.dumps(llm_output))

            # For each unique text in this chunk, apply decisions
            for item in chunk:
                rowidx = item["RowIndex"]
                decision = llm_decisions.get(rowidx, "EXCLUDE")  # default
                if decision == "KEEP":
                    # row remains keep => just append reason
                    for r in item["AllIndexes"]:
                        decision_reasons[r].append(
                            f"Kept by LLM for column '{col}'"
                        )
                else:
                    # row is excluded => mark final_decisions as EXCLUDE + reason
                    for r in item["AllIndexes"]:
                        final_decisions[r] = "EXCLUDE"
                        decision_reasons[r].append(
                            f"Excluded by LLM for column '{col}'"
                        )

    # C. After all columns, build the 'decision_reason' column in the original df
    df["decision_reason"] = df.index.map(
        lambda i: "; ".join(decision_reasons[i])
    )

    # D. Split into KEPT vs. EXCLUDED DataFrames
    kept_idx = [i for i, dec in final_decisions.items() if dec == "KEEP"]
    excluded_idx = [i for i, dec in final_decisions.items() if dec == "EXCLUDE"]

    filtered_df = df.loc[kept_idx].copy()
    excluded_df = df.loc[excluded_idx].copy()

    return filtered_df, excluded_df





### New functions to separate LLM processing from conceputal reasoning



import pandas as pd
import streamlit as st

def get_unique_column_values(df: pd.DataFrame, columns_to_check: list, debug: bool = False) -> list:
    unique_values = {}  # text_value -> {"RowIndexes": [...], "Text": str}
    for col in columns_to_check:
        for idx, value in df[col].items():
            if isinstance(value, (int, float)):
                value = str(value)
            if value not in unique_values:
                unique_values[value] = {"RowIndexes": [], "Text": value}
            unique_values[value]["RowIndexes"].append(idx)

    result = []
    for value, data in unique_values.items():
        # We'll pick the first index as representative, plus store the full list
        first_index = sorted(data["RowIndexes"])[0]
        result.append({
            "RowIndex": first_index,
            "Text": data["Text"],
            "AllIndexes": sorted(data["RowIndexes"])
        })

    if debug:
        print("Unique values result:", result)
    return result




def parse_llm_output(llm_output: str) -> dict:
    """
    Parses the JSON output from the LLM and returns a dictionary
    where the keys are 'RowIndex' and values are "KEEP" or "EXCLUDE".
    """
    try:
        data = json.loads(llm_output)
        if not isinstance(data, list): #Checks if data is a list to avoid errors.
          return {}
        result = {}
        for item in data:
          if isinstance(item, dict) and "RowIndex" in item and "Decision" in item:
            result[item["RowIndex"]] = item["Decision"]
        return result
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        print(f"Error parsing LLM output: {e}")
        return {} # Return an empty dict in case of error
    
    #######################################
# STEP 1: New function - filter_one_column
#######################################

from collections import defaultdict

def filter_one_column(
    df: pd.DataFrame,
    column_name: str,
    column_config: dict,
    chunk_size: int,
    model: str,
    temperature: float,
    reasoning_text: str,
    debug: bool = False
) -> pd.DataFrame:
    """
    Filters the given df by a single column using:
      - unique text extraction
      - exact-match auto keep
      - LLM-based conceptual keep/exclude
    Returns a *new* DataFrame with only the kept rows.
    """

    # 1) Get unique values in this single column
    #    reusing your existing function if it supports a single-col list
    unique_values_list = get_unique_column_values(df, [column_name], debug)

    # 2) We'll track final decisions + reasons per row index
    final_decisions = {}           # index -> "KEEP" or "EXCLUDE"
    decision_reasons = defaultdict(list)  # index -> list of reasons

    # 3) Gather user keywords for this column
    keywords_str = column_config.get("keywords", "")  # e.g. "CEO, Founder"
    user_keywords = set(kw.strip() for kw in keywords_str.split(",") if kw.strip())

    # 4) Separate exact matches (auto-KEEP) from those needing LLM
    to_send_to_llm = []
    for item in unique_values_list:
        text_value = item["Text"]
        if text_value in user_keywords:
            # Auto-keep all rows with this text
            for idx in item["AllIndexes"]:
                final_decisions[idx] = "KEEP"
                decision_reasons[idx].append(
                    f"Auto-KEEP (exact match) in column '{column_name}'"
                )
        else:
            to_send_to_llm.append(item)

    # 5) Chunk the remaining unique texts + call LLM
    chunked_unique_values = chunk_list(to_send_to_llm, chunk_size)

    for chunk in chunked_unique_values:
        # We only need the column's own keywords
        col_keywords = [kw.strip() for kw in keywords_str.split(",") if kw.strip()]

        # Build a prompt specifically for this column
        prompt = build_llm_prompt_single_col_json(
            row_summaries=[
                f'{{"RowIndex": {item["RowIndex"]}, "Text": "{item["Text"]}"}}'
                for item in chunk
            ],
            column_name=column_name,
            keywords=col_keywords,
            reasoning_text=reasoning_text,
            user_context=column_config.get("user_context", "")
        )

        if debug:
            print(f"\n== Prompt for column '{column_name}' ==\n{prompt}\n")

        llm_output = generate_text_with_function_call(
            prompt=prompt,
            model=model,
            temperature=temperature,
            debug=debug
        )

        if debug:
            print(f"\n== LLM output for '{column_name}' ==\n{llm_output}\n")

        llm_decisions = parse_llm_output(json.dumps(llm_output))

        # Apply LLM decisions
        for item in chunk:
            rowidx = item["RowIndex"]
            decision = llm_decisions.get(rowidx, "EXCLUDE")  # default if not found
            if decision == "KEEP":
                for idx in item["AllIndexes"]:
                    final_decisions[idx] = "KEEP"
                    decision_reasons[idx].append(
                        f"Kept by LLM for column '{column_name}'"
                    )
            else:
                for idx in item["AllIndexes"]:
                    final_decisions[idx] = "EXCLUDE"
                    decision_reasons[idx].append(
                        f"Excluded by LLM for column '{column_name}'"
                    )

    # 6) Any row not in final_decisions is excluded by default
    for idx in df.index:
        if idx not in final_decisions:
            final_decisions[idx] = "EXCLUDE"
            decision_reasons[idx].append(
                f"No LLM decision => EXCLUDE (column '{column_name}')"
            )

    # 7) Build a "decision_reason" column for debugging
    df["decision_reason"] = df.index.map(
        lambda i: "; ".join(decision_reasons[i])
    )

    # 8) Return only "KEEP" rows
    keep_idx = [i for i in df.index if final_decisions[i] == "KEEP"]
    return df.loc[keep_idx].copy()
