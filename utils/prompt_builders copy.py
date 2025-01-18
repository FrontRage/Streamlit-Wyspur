"""
prompt_builders.py

Provides various functions to build prompts for the LLM,
including both pipe-delimited and JSON output modes.
"""

def build_user_instructions(column_keywords_dict: dict) -> str:
    """
    Build a text block that tells the LLM what columns to look at
    and which keywords/concepts to exclude, using AND logic.
    """
    if not column_keywords_dict:
        return "No special exclude instructions provided."
    
    instructions = "Here are the columns and their filter keywords:\n"

    for col, keywords in column_keywords_dict.items():
        instructions += f"- Column '{col}': keep if it matches any of ({', '.join(keywords)})\n"

    return instructions


def build_conceptual_text(slider_value: int) -> str:
    """
    Return additional instructions for conceptual filtering based on a 1-5 scale:
    
        1 = Very strict
        2 = Moderate (leaning strict)
        3 = Moderate (balanced)
        4 = Moderate (leaning broad)
        5 = Very broad

    The slider_value argument can be an integer between 1 and 5 inclusive.
    This function lumps 2, 3, and 4 together into a single 'moderate' set of instructions by default,
    but you can split them further if needed.
    """
    if slider_value == 1:
        # LEVEL 1: EXTREMELY STRICT EXCLUSIONS
        return """
        LEVEL 1/5: EXTREMELY STRICT

        • Keep rows ONLY if they EXACTLY match the user’s keywords or a well-known,
          direct synonym.
        • Slight rewordings, minor variations, or related (but not identical) terms
          should NOT be kept.
        • Abbreviations are kept only if they are definitively the same concept
          (e.g., “CEO” vs. “C.E.O.”).
        • Location matching is applied strictly: keep ONLY if the column has an
          exact match to the country (or a universally recognized abbreviation).
          For instance, if “USA” is excluded, only exclude rows listing “USA” or
          “U.S.A.”, not a city in the USA unless it explicitly says “USA.”
        • If uncertain whether a term qualifies as an exact match or direct synonym,
          exclude the row (do NOT keep).
        """

    elif slider_value == 5:
        # LEVEL 5: VERY BROAD EXCLUSIONS
        return """
        **Guidelines for Broad Reasoning (Level 5/5)**:
        • Use **synonyms, tangential references, abbreviations**, or spelled-out forms:
        - E.g. "CEO" → "Chief Executive Officer," "CEOs," "C.E.O."
        - E.g. "VP" → "Vice President," "Senior VP," "V.P."
        • If the column references **locations** (e.g., "USA"), consider any **cities** or **regions** in that country as relevant.
        • For numeric **ranges** (e.g., "11 - 200 employees"), interpret partial or approximate references:
        - "50+" or "About 100 employees" might fit "11-200."
        • For partial overlaps in text (e.g., "CEOs" vs. "oceans"), assume it’s relevant if the overlap is plausible; if ambiguous, **KEEP**.
        • If you see *any* conceptual link—synonym, slight rewording, partial match—label it "KEEP."
        • If you are truly sure it doesn’t match any concept, "EXCLUDE."
        """
    else:
        # LEVELS 2, 3, AND 4: MODERATE EXCLUSIONS
        return f"""
        LEVEL {slider_value}/5: MODERATE APPROACH
        (Note: Levels 2, 3, and 4 belong to this 'moderate' category.)

        • Keep rows if they match or strongly relate to the user’s filter concepts.
        • Allow minor or faint references to remain; do NOT keep unless there is
          a clear, direct, or strong thematic connection.
        • Synonyms and abbreviations:
            - Keep if the synonym/abbreviation is well-known or commonly used.
            - If it is a subtle or obscure reference, do not keep.
        • Location handling:
            - If a country (or its standard abbreviation) is filtered, keep rows
              listing major cities of that country (e.g., if “USA” is excluded,
              exclude “New York,” “NY,” “Los Angeles,” “LA,” etc.).
            - For less obvious or ambiguous city references, only keep if you
              are certain they belong to the excluded country.
        • Contextual references:
            - If the row strongly implies the filtered concept, keep it.
            - If the connection is extremely tenuous or purely coincidental,
              exclude the row.
        • In cases of doubt, lean towards excluding the row (i.e., do NOT keep
          unless it is fairly certain to be related).
        """


def build_llm_prompt(
    row_summaries: list,
    min_idx: int,
    max_idx: int,
    user_instructions_text: str,
    reasoning_text: str,
    debug: bool
) -> str:
    """
    Centralize construction of the LLM prompt for a single chunk (pipe-delimited approach).
    If debug=False, we omit references to the Reason column to save tokens.

    This approach uses the older pipe-delimited format:
    RowIndex|Decision (|Reason)
    """
    system_instructions = f"""
    You are an expert data-cleaning assistant, acting as a conceptual filtering tool. 
    The user wants to Keep a row ONLY if it meets ALL the specified
    column-based keywords (AND-logic). If even one column does not match,
    you must EXCLUDE the row, unless the column that doesnt match doesnt have a value.

    IMPORTANT:
    - Do not include row indices that are not listed in the summaries.
    """

    if debug:
        format_instructions = """
        Return pipe-separated (|) text with three columns (no header):
        RowIndex|Decision|Reason
        """
    else:
        format_instructions = """
        Return pipe-separated (|) text with two columns (no header):
        RowIndex|Decision
        """

    prompt_for_llm = f"""
    {system_instructions}

    {user_instructions_text}

    Rows to be analyzed:

    {chr(10).join(row_summaries)}

    This is the reasoning you need to use to filter columns based on the provided keywords:
    {reasoning_text}

    Now that you have read and understood how broad you need to reason to keep rows
    based on the column provided filter keywords, go back analyze them and
    {format_instructions}
    """

    return prompt_for_llm.strip()


def build_llm_prompt_single_col(
    row_summaries: list,
    column_name: str,
    keywords: list,
    reasoning_text: str,
    debug: bool
) -> str:
    """
    BUILD A PIPE-DELIMITED PROMPT FOR ONE COLUMN
    (Older approach, kept for compatibility)

    Tells the LLM: If the text conceptually matches the keywords, label KEEP;
    else EXCLUDE. Output lines as: RowIndex|Decision (|Reason).
    """
    system_instructions = f"""
    You are a data-cleaning assistant focusing on one column: '{column_name}'.
    The user wants to KEEP rows if they conceptually match any of these keywords:
    {', '.join(keywords)}

    If a row does NOT match these keywords, label it EXCLUDE.

    IMPORTANT:
    1. You MUST output a line for EVERY RowIndex in the input, including those that do not match.
    2. If a row is not relevant or doesn't match, label it EXCLUDE.
    3. DO NOT skip or omit any RowIndex, even if it is completely unrelated.
    """

    if debug:
        format_instructions = """
        Return exactly one line per row in the format:
        RowIndex|Decision|Reason
        """
    else:
        format_instructions = """
        Return exactly one line per row in the format:
        RowIndex|Decision
        """

    summaries_text = "\n".join(row_summaries)

    prompt = f"""
    {system_instructions}

    Conceptual matching broadness:
    {reasoning_text}

    Here are the rows:
    {summaries_text}

    {format_instructions}
    """
    return prompt.strip()


def build_llm_prompt_single_col_json(
    row_summaries: list,
    column_name: str,
    keywords: list,
    reasoning_text: str,
    debug: bool
) -> str:
    """
    BUILD A JSON-BASED PROMPT FOR ONE COLUMN

    Instructs the LLM to return STRICT JSON for each row:
      - "RowIndex": <integer>
      - "Decision": "KEEP" or "EXCLUDE"
      - "Reason": optional if debug=False (omitted), or included if debug=True

    This helps parse the output with json.loads, reducing errors vs. pipe-delimited.
    """
    system_instructions = f"""
    You are an advanced LLM conceptual reasoning filter, focusing on **one column**: "{column_name}".
    Traditional filters cannot catch synonyms, slight variations, ranges, or tangential connections, but you can!

    **Context for This Column**:
    - You are analyzing "{column_name}" entries in a spreadsheet. 
    - Use your broader knowledge and reasoning about how data might be represented in this column to interpret potential synonyms, abbreviations, numeric ranges, or other variations.

    **Your Task**:
    You will be given a series of row texts (one for each row). For each row:
    1) Compare the text to the following filter concepts (broadly and creatively):  
    {', '.join(keywords)}
    2) Decide whether this row **thematically matches** (i.e., is relevant to) **any** of these concepts. 
    - If it matches (even loosely or by synonyms/associated ideas), label this row "KEEP."
    - Otherwise, label it "EXCLUDE."

    **Important Requirements**:
    1. Return an entry for **every** row (do not skip or omit rows).
    2. If a row’s text **cannot** be related to any of the user’s keywords, label it "EXCLUDE."
    3. If you are uncertain, **lean** toward "KEEP" (err on the side of over-filtering).
    4. Produce **only valid JSON** (an array of objects) with no extra commentary or markdown. 
    """

    if debug:
        # Debug => We want a "Reason" field in each object for clarity
        example_output = """
        Example valid JSON output (with Reason):
        [
          {
            "RowIndex": 301,
            "Decision": "KEEP",
            "Reason": "Text 'CEO' is a variation of a filtered keyword."
          },
          {
            "RowIndex": 302,
            "Decision": "EXCLUDE",
            "Reason": "No match to any keywords."
          }
        ]
        """
        debug_notes = """
        For each row, create an object with keys:
          "RowIndex" (integer),
          "Decision" ("KEEP" or "EXCLUDE"),
          "Reason" (short explanation).
        """
    else:
        # Non-debug => we omit the "Reason" field entirely to minimize token usage
        example_output = """
        Example valid JSON output (no Reason):
        [
          {
            "RowIndex": 301,
            "Decision": "KEEP"
          },
          {
            "RowIndex": 302,
            "Decision": "EXCLUDE"
          }
        ]
        """
        debug_notes = """
        For each row, create an object with keys:
          "RowIndex" (integer),
          "Decision" ("KEEP" or "EXCLUDE").
        No "Reason" field is required or allowed in this mode.
        """

    summaries_text = "\n".join(row_summaries)

    prompt = f"""
    {system_instructions}

    {reasoning_text}

    {example_output}

    {debug_notes}

    Return ONLY this JSON array (no markdown or extra text).

    Now, analyze the following rows (each has RowIndex, Text) and output your final JSON array:
    {summaries_text}
    """

    return prompt.strip()
