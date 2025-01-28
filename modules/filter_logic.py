import streamlit as st
import pandas as pd
import sys
from io import StringIO
import json
import re
from time import sleep
from thefuzz import fuzz

# We'll import from our utils modules:
from utils.chunking import chunk_dataframe
from utils.df_helpers import make_chunk_summaries
from utils.prompt_builders import build_llm_prompt_single_col_json


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




def filter_df_via_llm_per_column(
    df: pd.DataFrame,
    columns_to_check: list,
    filter_column_config: dict,
    chunk_size: int,
    reasoning_text: str,
    model: str,
    temperature: float,
    debug: bool = False
) -> dict:
    """
    Step 2 (JSON-Based Version): For each column, we do a separate LLM call per chunk.
    The LLM should return STRICT JSON output (one array of objects per prompt).
    Return decisions_per_column = {
      colName: { row_idx: "KEEP"/"EXCLUDE", ... },
      ...
    }

    Parameters
    ----------
    df : pd.DataFrame
        The full DataFrame to filter.
    columns_to_check : list of str
        Column names the user wants to filter on.
    filter_column_config : dict
        A dict containing filter config for each column, including logic and keywords.
    chunk_size : int
        Number of rows per chunk to send in one prompt.
    reasoning_text : str
        Conceptual instructions (e.g. from the slider).
    model : str
        Which LLM model to call (e.g., "gpt-3.5-turbo").
    temperature : float
        Sampling temperature for the LLM call.
    debug : bool, optional
        If True, we show prompt/response expansions and parse details.

    Returns
    -------
    dict
        A nested dict mapping each column -> {row_idx -> "KEEP" or "EXCLUDE"}.

    Notes
    -----
    - Replaces the old pipe-delimited approach with a JSON approach for more reliable parsing.
    - Uses build_llm_prompt_single_col_json to build the prompt and
      parse_llm_decisions_single_col_json to parse the returned JSON.
    - For row-level decisions, see aggregate_column_decisions() and filter_df_by_aggregated_decisions().
    """
    import logging
    from thefuzz import fuzz # <---- IMPORT FUZZ HERE - IMPORTANT!

    logging.basicConfig(
        filename="llm_debug.log",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    decisions_per_column = {}
    if df.empty:
        return decisions_per_column

    total_rows = len(df)
    total_chunks = (total_rows // chunk_size) + (1 if total_rows % chunk_size else 0)

    progress_bar = st.progress(0)
    chunks_done = 0

    # Optional debug container to avoid clutter
    debug_container = st.container() if debug else None # <----- Fixed typo: st.container.container() -> st.container()

    # We'll chunk the DataFrame
    for chunk_idx, chunk_df in enumerate(chunk_dataframe(df, chunk_size), start=1):
        valid_indices = set(chunk_df.index)

        # Show progress in debug
        if debug_container:
            with debug_container:
                st.markdown(f"### Debug: Chunk {chunk_idx}/{total_chunks}")

        # For each column the user wants to check
        for col in columns_to_check:
            # --- Get column-specific config from filter_column_config ---
            col_config = filter_column_config.get(col, {}) # Get config for this column
            keywords_str = col_config.get('keywords', "") # Get keywords as a STRING
            logic = col_config.get('logic', "Conceptual Reasoning") # Get logic, default to Conceptual Reasoning
            user_context = col_config.get('user_context', "") # Get user context (if any)

            # --- SPLIT KEYWORD STRING INTO A LIST HERE ---
            keywords_list = [kw.strip() for kw in keywords_str.split(',')] # Split by comma, strip whitespace
            # Now 'keywords_list' is a list of strings (e.g., ['CEO', 'VP of Product', 'Director'])

            st.write(f"--- Debug: Column: {col}, Logic: {logic}") # <--- DEBUG PRINT: LOGIC
            st.write(f"--- Debug: Column: {col}, Keywords: {keywords_list}") # <--- DEBUG PRINT: KEYWORDS
            st.write(f"--- Debug: Column: {col}, User Context: {user_context}") # <--- DEBUG PRINT: USER CONTEXT

            if not keywords_list: # Use the LIST of keywords
                continue  # No keywords => skip this column


            # --- CONDITIONAL LOGIC BASED ON 'logic' SETTING ---
            if logic == "Conceptual Reasoning":
                st.write(f"--- Debug: Column: {col}, Applying Conceptual Reasoning Logic (LLM will be called)") # <--- Placeholder message
                # ... (Conceptual Reasoning logic will go here in future steps) ...
                # For now, just proceed with the existing prompt building and LLM call (for Conceptual Reasoning)

            elif logic == "Exact Match":
                st.write(f"--- Debug: Column: {col}, Applying Exact Match Logic") # Debug message
                col_decisions = {} # Initialize decisions for this column

                for row_idx, row_data in chunk_df.iterrows():
                    cell_value = str(row_data[col]).lower()
                    is_match = False # Assume no match initially

                    for keyword in keywords_list:
                        if cell_value == keyword.lower().strip(): # Exact match (case-insensitive, whitespace stripped)
                            is_match = True
                            break # Exit keyword loop if exact match found
                        elif fuzz.ratio(cell_value, keyword.lower().strip()) >= 85: # Fuzzy match (threshold 85)
                            is_match = True
                            break # Exit keyword loop if fuzzy match found

                    # --- REVERSED KEEP/EXCLUDE LOGIC HERE ---
                    if is_match:
                        col_decisions[row_idx] = "KEEP" # <--- CORRECTED: KEEP if exact or fuzzy match
                    else:
                        col_decisions[row_idx] = "EXCLUDE" # <--- CORRECTED: EXCLUDE if NOT exact or fuzzy match

                decisions_per_column[col] = col_decisions # Store decisions for this column
                continue # <--- IMPORTANT: Skip LLM call for "Exact Match" - CONTINUE TO NEXT COLUMN

            elif logic == "User Context":
                st.write(f"--- Debug: Column: {col}, Applying User Context Logic (Placeholder - LLM will be called)") # Debug message
                # ... (User Context logic will go here in Phase 4.4) ...
                # For now, fall through to Conceptual Reasoning (LLM call below)

            else: # Conceptual Reasoning (and default case)
                st.warning(f"Unknown logic type '{logic}' for column '{col}'. Defaulting to Conceptual Reasoning.")
                st.write(f"--- Debug: Column: {col}, Applying Conceptual Reasoning Logic (Default - LLM will be called)") # <--- Placeholder message
                # ... (Conceptual Reasoning logic will go here in future steps) ...
                # Fall through to prompt building and LLM call below


            # 1) Build row summaries for JUST this column
            row_summaries = []
            for row_idx, row_data in chunk_df.iterrows():
                text_for_llm = str(row_data[col])
                summary = f"RowIndex={row_idx}|Column='{col}'|Text='{text_for_llm}'"
                row_summaries.append(summary)

            # 2) Build a JSON-based prompt
            prompt = build_llm_prompt_single_col_json(
                row_summaries=row_summaries,
                column_name=col,
                keywords=keywords_list, # <--- Pass the LIST of keywords here
                reasoning_text=reasoning_text,
                user_context=user_context # <--- NEW: Pass user_context here!
            )

            # Debug: Show the prompt
            if debug_container:
                with debug_container.expander(f"Prompt for Column '{col}', Chunk {chunk_idx} (JSON)", expanded=False):
                    st.code(prompt, language="markdown")

            # Generate the LLM response
            try:
                llm_response = generate_text_with_function_call(
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    debug=debug
                )

                # Debug: Show the structured JSON response
                if debug_container:
                    with debug_container.expander(f"LLM JSON Response for Column '{col}', Chunk {chunk_idx} (Structured JSON)", expanded=False):
                        st.json(llm_response)

                # Parse JSON decisions
                col_decisions = parse_llm_decisions_single_col_json(
                    llm_response=llm_response,
                    valid_indices=valid_indices,
                    debug=debug
                )

            except Exception as e:
                error_message = f"Failed to parse LLM response for Column '{col}', Chunk {chunk_idx}: {e}"
                st.error(error_message)
                logging.error(error_message)

                st.json(llm_response)  # Always show the raw JSON response
                continue

            # Debug: Show parsed decisions
            if debug_container:
                with debug_container.expander(f"Parsed JSON Decisions for Column '{col}', Chunk {chunk_idx}", expanded=False):
                    st.write(col_decisions)

            # Update decisions_per_column
            if col not in decisions_per_column:
                decisions_per_column[col] = {}
            decisions_per_column[col].update(col_decisions)

        # Update progress bar
        chunks_done += 1
        progress_bar.progress(chunks_done / total_chunks)

    return decisions_per_column





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
    Given the row-level decisions, return a new DataFrame
    containing only rows labeled 'KEEP'.
    """
    keep_indices = [idx for idx, dec in final_decisions.items() if dec == "KEEP"]
    filtered_df = df.loc[keep_indices]
    return filtered_df

def filter_df_master(
    df: pd.DataFrame,
    columns_to_check: list,
    filter_column_config: dict,  # <--- CORRECT: 'filter_column_config' in THIRD position
    chunk_size: int,          # <--- CORRECT: 'chunk_size' ADDED, in FOURTH position
    model: str,
    temperature: float,
    reasoning_text: str,
    debug: bool = False
) -> pd.DataFrame:
    """
    High-level function that:
      1) Calls filter_df_via_llm_per_column(...) to get per-column decisions
      2) Aggregates those decisions into row-level keep/exclude (and collects which columns caused exclude)
      3) Stores those exclusion reasons in a 'exclusion_reason' column
      4) Returns the filtered DataFrame (containing only rows labeled 'KEEP')
    """
    # Step 2: column-level decisions
    decisions_per_col = filter_df_via_llm_per_column(
        df=df,
        columns_to_check=columns_to_check,
        filter_column_config=filter_column_config, # <--- Correctly passing 'filter_column_config' NOW
        chunk_size=chunk_size,         # <--- Correctly passing 'chunk_size'
        model=model,
        temperature=temperature,
        reasoning_text=reasoning_text,
        debug=debug
    )

    # Step 3: aggregate decisions => get final_decisions + columns that triggered EXCLUDE
    # NOTE: We now expect aggregate_column_decisions to return (final_decisions, exclusion_reasons)
    final_decisions, exclusion_reasons = aggregate_column_decisions(
        decisions_per_col,
        debug=debug
    )

    # Store the exclusion reasons for each row in a debug column
    # (rows that remain 'KEEP' will have an empty string, i.e. no excluded columns).
    df["exclusion_reason"] = df.index.map(
        lambda idx: ", ".join(exclusion_reasons.get(idx, []))
    )

    # Use the final decisions to filter the DataFrame to only 'KEEP' rows
    filtered_df = filter_df_by_aggregated_decisions(df, final_decisions)
    return filtered_df
