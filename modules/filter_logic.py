import streamlit as st
import pandas as pd
import sys
from io import StringIO

# We'll import from our utils modules:
from utils.chunking import chunk_dataframe
from utils.df_helpers import make_chunk_summaries
from utils.prompt_builders import build_llm_prompt

# We'll import our OpenAI function from openai_module
from modules.openai_module import generate_text_basic


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
