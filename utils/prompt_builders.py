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
    Return instructions for conceptual filtering based on a 3-level scale:

        1 = Strict
        2 = Moderate
        3 = Broad

    Now with rephrased numeric range instructions (Option 1) in Moderate and Broad levels.
    """
    if slider_value == 1:
        # LEVEL 1: STRICT - CONCEPTUAL REASONING (3-LEVEL SLIDER)
        return """
        **LEVEL: STRICT - CONCEPTUAL FILTERING MODIFIER**
        - Apply a strict interpretation of the keywords, **only** keeping exact matches and direct synonyms.
        """

    elif slider_value == 3:
        # LEVEL 3: BROAD - CONCEPTUAL FILTERING (3-LEVEL SLIDER) - NUMERIC RANGE INSTRUCTIONS UPDATED (OPTION 1)
        return """
       **LEVEL: BROAD - CONCEPTUAL FILTERING MODIFIER**
       -  Apply a broad interpretation of the keywords, and keep any text that shows a plausible conceptual link.
        """
    else: # slider_value == 2: MODERATE - CONCEPTUAL REASONING (3-LEVEL SLIDER) - NUMERIC RANGE INSTRUCTIONS UPDATED (OPTION 1)
        return f"""
        **LEVEL: MODERATE - CONCEPTUAL FILTERING MODIFIER**
        - Apply a moderate interpretation of the keywords, and keep any text that has a clear connection with the keywords.
        """

def build_llm_prompt_single_col_json(
    row_summaries: list,
    column_name: str,
    keywords: list,
    reasoning_text: str,
    user_context: str = ""
) -> str:
    """
    BUILD A JSON-BASED PROMPT FOR ONE COLUMN (Column-Specific & with User Context)

    Instructs the LLM to analyze rows for a specific column and return JSON-based decisions:
      - "RowIndex": <integer>
      - "Decision": "KEEP" or "EXCLUDE"

    Now with enhanced column-specific instructions and optional user context.
    """
    system_instructions = f"""
    You are a highly specialized AI filter, designed for **conceptual reasoning on spreadsheet data, one column at a time.**
    You are now focusing EXCLUSIVELY on the **"{column_name}" column**.
    Your goal is to filter rows based on whether the **"{column_name}" values** conceptually relate to user-provided keywords.

    **Detailed Context for "{column_name}" Column Analysis**:
    - You are examining entries specifically within the **"{column_name}" column** of a spreadsheet.
    - Understand that spreadsheet columns can contain various types of textual data, numbers, or locations.
    - Apply broad, conceptual reasoning to identify matches, considering synonyms, related concepts, and variations typical for **data in spreadsheet columns.**

    **Filter Keywords for "{column_name}"**:
    The user wants to filter (keep) rows that are conceptually related to *any* of these concepts:
    {', '.join(keywords)}

    **IMPORTANT: Prioritize User-Provided Context**
    - The following user-provided context is **paramount**. You must prioritize it above all other instructions and keywords. If the user context contradicts any other instruction, **follow the user context instruction**.

    """
    if user_context:
        system_instructions += f"""

    **Additional User-Provided Context/Instructions for "{column_name}" Filtering:**
    {user_context}

    **Instructions for Inclusion/Exclusion**:
    - **Keep** rows that are related to the keywords and the user-provided context unless there is a clear contradiction.
    - **Exclude** rows if they clearly and directly contradict both the keywords and the user-provided context.
    - If a title includes words that directly contradict the user context, or has characteristics from a role that is explicitly defined as undesirable in the context, then it **MUST** be excluded.
    - For example, if the user context specifies to only keep "Marketing" titles, exclude titles containing "Sales" or "Business Development" keywords and vice versa.
    - **Do not** include roles that are outside the specified user context, even if it contains keywords that match.
    - **Do not** exclude roles that align with the specified user context, even if they do not match the keywords directly.
     - When in doubt, if the text has a plausible link with the context, then keep it.
    """
    else:
        system_instructions += f"""
    **Instructions for Inclusion/Exclusion**:
        - When no user context is provided, **keep** rows that conceptually match the provided keywords.
        - When no user context is provided, **exclude** rows that do not clearly relate to the provided keywords.
        - In cases of doubt, lean towards keeping the row (i.e., do NOT exclude unless it is fairly certain to be unrelated).
    """


    system_instructions += f"""
    **Your Task (Column-Specific Conceptual Filtering)**:
    For each row, analyze the **"{column_name}" value** and decide:
        1) Does it conceptually relate to *any* of the provided filter concepts for the **"{column_name}" column** AND aligns with the provided user context (if any)?
       * Consider the general context of data in spreadsheet columns.
       * Use synonyms, abbreviations, and related terms, **but only if it aligns with the user-provided context (if any)**.
       * Apply the conceptual reasoning guidelines provided below (in 'Reasoning Text').
       2) Based on your conceptual understanding AND the user-provided context (if any), label each row as:
       - "KEEP": If the **"{column_name}" value** is conceptually relevant to the keywords AND aligns with the user context (if any), unless there is a clear contradiction.
       - "EXCLUDE": If the **"{column_name}" value** is clearly NOT conceptually relevant to the keywords, or directly contradicts the user context (if any).

    Remember, you are filtering rows based on conceptual relevance to keywords **specifically within the "{column_name}" column** AND with the **user-provided context as the most important factor (if any).**
    """

    summaries_text = "\n".join(row_summaries)

    prompt = f"""
    {system_instructions}

    {reasoning_text}

    Now, analyze the following rows (each has RowIndex, Text) and output your final JSON array:
    {summaries_text}
    """

    return prompt.strip()