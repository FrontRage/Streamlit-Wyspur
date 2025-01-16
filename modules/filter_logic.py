import streamlit as st
import pandas as pd
import sys
from io import StringIO

# We'll import from our utils modules:
from utils.chunking import chunk_dataframe
from utils.df_helpers import make_chunk_summaries
from utils.prompt_builders import build_llm_prompt
from utils.prompt_builders import build_llm_prompt_single_col


# We'll import our OpenAI function from openai_module
from modules.openai_module import generate_text_basic

def parse_llm_decisions_single_col(
    llm_response: str,
    valid_indices: set,
    debug: bool = False
) -> dict:
    """
    Parse the LLM's output for a single column chunk:
    - We expect either "RowIndex|Decision" or "RowIndex|Decision|Reason".
    - Return { row_idx: "KEEP" or "EXCLUDE" } for all valid row indices.
    """

    decisions = {}

    try:
        if debug:
            # 3 columns
            response_df = pd.read_csv(
                StringIO(llm_response),
                sep="|",
                header=None,
                names=["RowIndex", "Decision", "Reason"]
            )
        else:
            # 2 columns
            response_df = pd.read_csv(
                StringIO(llm_response),
                sep="|",
                header=None,
                names=["RowIndex", "Decision"]
            )
            response_df["Reason"] = ""

    except Exception as e:
        st.error(f"Error parsing LLM response: {e}")
        st.write("Raw LLM response:")
        st.write(llm_response)
        return decisions

    for _, row in response_df.iterrows():
        try:
            idx = int(row["RowIndex"])
            if idx not in valid_indices:
                continue
            decision_str = str(row["Decision"]).strip().upper()
            if decision_str not in ["KEEP", "EXCLUDE"]:
                # If the LLM response is unexpected, skip or handle
                continue

            decisions[idx] = decision_str
        except ValueError:
            # row index not integer, skip
            continue

    return decisions


def parse_llm_decisions(llm_response: str, valid_indices: set, debug: bool = False) -> dict:
    """
    Parse the LLM's pipe-delimited response into a dictionary of 
    row_index -> (Decision, Reason).

    If debug=False, we expect the format: RowIndex|Decision
    If debug=True, we expect the format:  RowIndex|Decision|Reason
    
    Returns: 
      - A dict mapping each row_index to a tuple (Decision, Reason).
        e.g. { 0: ("KEEP", "Explanation..."), 1: ("EXCLUDE", "Another reason...") }
    """
    decisions = {}

    try:
        if debug:
            # Expect three columns: RowIndex, Decision, Reason
            response_df = pd.read_csv(
                StringIO(llm_response),
                sep="|",
                header=None,
                names=["RowIndex", "Decision", "Reason"]
            )
        else:
            # Expect two columns: RowIndex, Decision
            response_df = pd.read_csv(
                StringIO(llm_response),
                sep="|",
                header=None,
                names=["RowIndex", "Decision"]
            )
            response_df["Reason"] = ""  # Fill in a blank reason column if debug=False
    except Exception as e:
        st.error(f"Error parsing LLM response: {e}")
        st.write("Raw LLM response:")
        st.write(llm_response)
        return {}

    for _, row in response_df.iterrows():
        try:
            row_index = int(row["RowIndex"])
            decision = str(row["Decision"]).strip().upper()
            reason = str(row["Reason"]) if pd.notnull(row["Reason"]) else ""

            # Only store the decision if the RowIndex is valid for this chunk
            if row_index in valid_indices:
                decisions[row_index] = (decision, reason)
        except ValueError:
            # If RowIndex is not a valid integer, skip
            continue

    return decisions


def filter_df_via_llm_summaries(
    df: pd.DataFrame,
    user_instructions_text: str,
    columns_to_summarize: list,
    chunk_size: int = 20,
    conceptual_slider: int = 5,
    reasoning_text: str = "",  # <-- Added this param to inject conceptual text
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.0,
    top_p: float = 1.0,
    debug: bool = False,
    max_debug_display: int = 50
) -> pd.DataFrame:
    """
    Filters the DataFrame using an LLM-based conceptual exclusion approach.

    Steps:
      1. Chunk the DataFrame into smaller pieces.
      2. Build summaries for each row in the chunk (for context).
      3. Construct an LLM prompt with instructions + row summaries (+ conceptual reasoning).
      4. Parse the LLM's decisions (KEEP or EXCLUDE).
      5. Combine decisions from all chunks and create the filtered DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        user_instructions_text (str): The AND-logic instructions for columns/keywords.
        columns_to_summarize (list): List of columns to summarize in the prompt.
        chunk_size (int): Number of rows per chunk (default 200).
        conceptual_slider (int): Slider value (1 to 5) for conceptual strictness.
        reasoning_text (str): Extra conceptual instructions from build_conceptual_text().
        model (str): OpenAI model name (default "gpt-3.5-turbo").
        temperature (float): Sampling temperature (0..2).
        top_p (float): Nucleus sampling (0..1).
        debug (bool): If True, includes the Reason column + displays LLM prompts/responses.
        max_debug_display (int): How many decisions to display in debug summary.

    Returns:
        pd.DataFrame: The filtered DataFrame, containing only rows kept by the LLM.
    """

    decisions = {}
    total_rows = len(df)
    if total_rows == 0:
        # Edge case: empty dataframe
        return df

    # Calculate how many chunks in total
    total_chunks = (total_rows // chunk_size) + (1 if total_rows % chunk_size else 0)
    progress_bar = st.progress(0)
    current_chunk = 0

    for chunk_idx, chunk_df in enumerate(chunk_dataframe(df, chunk_size)):
        current_chunk += 1

        # Prepare a set of valid indices for this chunk
        valid_indices_set = set(chunk_df.index)

        # Summaries for each row in the chunk
        row_summaries = make_chunk_summaries(chunk_df, columns_to_summarize)

        # Build the LLM prompt
        prompt_for_llm = build_llm_prompt(
            row_summaries=row_summaries,
            min_idx=chunk_df.index.min(),
            max_idx=chunk_df.index.max(),
            user_instructions_text=user_instructions_text,
            reasoning_text=reasoning_text,  # <-- Now we actually pass your conceptual text
            debug=debug
        )

        # Call the LLM via our generate_text_basic function
        llm_response = generate_text_basic(
            prompt_for_llm,
            model=model,
            temperature=temperature,
            top_p=top_p
        )

        # Debug output
        if debug:
            st.markdown(f"### Debug Info for Chunk {chunk_idx}")
            with st.expander("LLM Prompt"):
                st.write(prompt_for_llm)
            with st.expander("LLM Raw Response"):
                st.write(llm_response)

        # Parse decisions from the LLM's response
        chunk_decisions = parse_llm_decisions(llm_response, valid_indices_set, debug=debug)
        decisions.update(chunk_decisions)

        # Update progress bar
        progress_fraction = current_chunk / total_chunks
        progress_bar.progress(progress_fraction)

    # Final decisions: keep only rows that are "KEEP"
    keep_indices = [idx for idx, (dec, _) in decisions.items() if dec == "KEEP"]
    filtered_df = df.loc[keep_indices]

    # (Optional) Display debug summary of decisions
    if debug and max_debug_display > 0:
        st.subheader("LLM Debug Decisions (Sample)")
        display_count = 0
        for idx, (dec, rsn) in decisions.items():
            st.write(f"Row {idx}: {dec} | Reason: {rsn}")
            display_count += 1
            if display_count >= max_debug_display:
                st.write(f"(Stopped after {max_debug_display} rows...)")
                break

    return filtered_df

def filter_df_via_llm_per_column(
    df: pd.DataFrame,
    columns_to_check: list,
    column_keywords: dict,
    chunk_size: int,
    reasoning_text: str,
    model: str,
    temperature: float,
    debug: bool = False
) -> dict:
    """
    Step 2: For each column, we do a separate LLM call per chunk.
    Return decisions_per_column = {
      colName: { row_idx: "KEEP"/"EXCLUDE", ... },
      ...
    }
    """
    decisions_per_column = {}
    if df.empty:
        return decisions_per_column

    total_rows = len(df)
    total_chunks = (total_rows // chunk_size) + (1 if total_rows % chunk_size else 0)
    progress_bar = st.progress(0)
    chunks_done = 0

    # We'll use an expanding container for debug logs, so they don't clutter the interface.
    debug_container = None
    if debug:
        debug_container = st.container()  # We'll write logs here

    for chunk_idx, chunk_df in enumerate(chunk_dataframe(df, chunk_size), start=1):
        valid_indices = set(chunk_df.index)

        if debug and debug_container:
            with debug_container:
                st.markdown(f"### Debug: Chunk {chunk_idx}/{total_chunks}")

        for col in columns_to_check:
            if col not in column_keywords:
                continue  # no keywords -> skip

            # build row summaries for JUST this column
            row_summaries = []
            for row_idx, row_data in chunk_df.iterrows():
                text_for_llm = str(row_data[col])
                summary = f"RowIndex={row_idx}|Column='{col}'|Text='{text_for_llm}'"
                row_summaries.append(summary)

            prompt = build_llm_prompt_single_col(
                row_summaries=row_summaries,
                column_name=col,
                keywords=column_keywords[col],
                reasoning_text=reasoning_text,
                debug=debug
            )

            # --- DEBUG: Show the prompt
            if debug and debug_container:
                with debug_container.expander(f"Prompt for Column '{col}', Chunk {chunk_idx}", expanded=False):
                    st.code(prompt, language="markdown")  

            # call LLM
            llm_response = generate_text_basic(
                prompt,
                model=model,
                temperature=temperature
            )

            # --- DEBUG: Show the raw response
            if debug and debug_container:
                with debug_container.expander(f"LLM Response for Column '{col}', Chunk {chunk_idx}", expanded=False):
                    st.code(llm_response, language="markdown")  

            # parse decisions
            col_decisions = parse_llm_decisions_single_col(
                llm_response=llm_response,
                valid_indices=valid_indices,
                debug=debug
            )
            
            if debug and debug_container:
                with debug_container.expander(
                    f"Parsed Decisions for Column '{col}', Chunk {chunk_idx}", expanded=False
                ):
                    st.write(col_decisions)

            
            # update decisions_per_column
            if col not in decisions_per_column:
                decisions_per_column[col] = {}
            decisions_per_column[col].update(col_decisions)

        chunks_done += 1
        progress_bar.progress(chunks_done / total_chunks)

    return decisions_per_column


def aggregate_column_decisions(decisions_per_column: dict, debug: bool = False) -> dict:
    """
    Given a dictionary of the form:
        {
          "ColumnA": {0: "KEEP", 1: "EXCLUDE", ...},
          "ColumnB": {0: "KEEP", 1: "KEEP",    ...},
          ...
        }
    Return a final row-level dictionary:
        final_decisions[row_index] = "KEEP" or "EXCLUDE"

    Logic: row_index is "KEEP" only if *all* columns say "KEEP" for that row.
    If a row doesn't appear in a particular column's dictionary, treat that as "EXCLUDE" or skip
    based on your logic (usually "EXCLUDE" by default).
    """
    from collections import defaultdict

    final_decisions = {}
    # Gather all row indices that appear in *any* column's decision dict
    all_row_indices = set()
    for col_name, decision_map in decisions_per_column.items():
        all_row_indices.update(decision_map.keys())

    for row_idx in all_row_indices:
        # We'll check each column's decision for row_idx
        keep_for_all_columns = True  # assume True, then check for any EXCLUDE
        for col_name, decision_map in decisions_per_column.items():
            # if row_idx is missing from decision_map, you can treat as "EXCLUDE"
            # or interpret it differently. Typically, we do EXCLUDE to be safe.
            if row_idx not in decision_map:
                keep_for_all_columns = False
                break

            decision_for_this_col = decision_map[row_idx]
            if decision_for_this_col == "EXCLUDE":
                keep_for_all_columns = False
                break

        if keep_for_all_columns:
            final_decisions[row_idx] = "KEEP"
        else:
            final_decisions[row_idx] = "EXCLUDE"

    # --------------------------
    # DEBUG: Show final decisions, if debug=True
    # --------------------------
    if debug:
        import streamlit as st
        with st.expander("Final Aggregated Row Decisions (Debug)", expanded=False):
            st.write(final_decisions)

    return final_decisions


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
    column_keywords: dict,
    chunk_size: int,
    model: str,
    temperature: float,
    reasoning_text: str,
    debug: bool = False
) -> pd.DataFrame:
    """
    High-level function that:
      1) Calls filter_df_via_llm_per_column(...) to get per-column decisions
      2) Aggregates those decisions into row-level keep/exclude
      3) Returns the filtered DataFrame
    """
    # Step 2: column-level decisions
    decisions_per_col = filter_df_via_llm_per_column(
        df=df,
        columns_to_check=columns_to_check,
        column_keywords=column_keywords,
        chunk_size=chunk_size,
        model=model,
        temperature=temperature,
        reasoning_text=reasoning_text,
        debug=debug
    )

    # Step 3: aggregate to get final row decisions, passing debug as well
    final_decisions = aggregate_column_decisions(decisions_per_col, debug=debug)

    # Use the final decisions to filter the DataFrame
    filtered_df = filter_df_by_aggregated_decisions(df, final_decisions)
    return filtered_df
