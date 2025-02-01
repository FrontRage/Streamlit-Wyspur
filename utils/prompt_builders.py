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


# def build_conceptual_text(slider_value: int) -> str:
#     """
#     Return instructions for conceptual filtering based on a 3-level scale:

#         1 = Strict
#         2 = Moderate
#         3 = Broad

#     Now with rephrased numeric range instructions (Option 1) in Moderate and Broad levels.
#     """
#     if slider_value == 1:
#         # LEVEL 1: STRICT - CONCEPTUAL REASONING (3-LEVEL SLIDER)
#         return """
#         **LEVEL: STRICT - CONCEPTUAL FILTERING MODIFIER**
#         - Apply a strict interpretation of the keywords, **only** keeping exact matches and direct synonyms.
#         """

#     elif slider_value == 3:
#         # LEVEL 3: BROAD - CONCEPTUAL FILTERING (3-LEVEL SLIDER) - NUMERIC RANGE INSTRUCTIONS UPDATED (OPTION 1)
#         return """
#        **LEVEL: BROAD - CONCEPTUAL FILTERING MODIFIER**
#        -  Apply a broad interpretation of the keywords, and keep any text that shows a plausible conceptual link.
#         """
#     else: # slider_value == 2: MODERATE - CONCEPTUAL REASONING (3-LEVEL SLIDER) - NUMERIC RANGE INSTRUCTIONS UPDATED (OPTION 1)
#         return f"""
#         **LEVEL: MODERATE - CONCEPTUAL FILTERING MODIFIER**
#         - Apply a moderate interpretation of the keywords, and keep any text that has a clear connection with the keywords.
#         """

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
        LEVEL: STRICT - CONCEPTUAL FILTERING

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

    elif slider_value == 3:
        # LEVEL 3: BROAD - CONCEPTUAL FILTERING (3-LEVEL SLIDER) - NUMERIC RANGE INSTRUCTIONS UPDATED (OPTION 1)
        return """
        **LEVEL: BROAD - CONCEPTUAL FILTERING**

        • Use **synonyms, abbreviations**, or spelled-out forms:
        - E.g. "CEO" → "Chief Executive Officer," "CEOs," "C.E.O." "VP" → "Vice President," "Senior VP," "V.P."
        - If "VP of sales" is provided, Keep "SVP of Sales" or variations of that tile, but not a "VP of Product" as this title works in a different department.
        • If the column references **locations** (e.g., "USA"), consider any **cities** or **regions** in that country as relevant.
        • For numeric **employee count ranges** (e.g., "11 - 200 employees"):  **Keep rows if the listed employee count range *overlaps or falls within* the user-provided range.**
            - Examples: If user provides "11 - 200", keep "50-200", "100-200", "11-100", "11-500", "150", but exclude "1-10", "300-500", "5".
        • For partial overlaps in text (e.g., "CEOs" vs. "oceans"), assume it’s relevant if the overlap is plausible; if ambiguous, **EXCLUDE**.
        • If you see *any* conceptual link—synonym, slight rewording,label it "KEEP."
        • If you are truly sure it doesn’t match any concept, "EXCLUDE."
        """
    else: # slider_value == 2: MODERATE - CONCEPTUAL REASONING (3-LEVEL SLIDER) - NUMERIC RANGE INSTRUCTIONS UPDATED (OPTION 1)
        return f"""
        LEVEL: MODERATE - CONCEPTUAL FILTERING

        • Keep rows if they match or strongly relate to the user’s filter concepts.
        • Allow minor or faint references to remain; do NOT keep unless there is
          a clear, direct, or strong thematic connection.
        • Synonyms and abbreviations:
            - Keep if the synonym/abbreviation is well-known or commonly used.
            - If it is a subtle or obscure reference, do not keep.
        • Location handling:
            - If a country (or its standard abbreviation) is filtered, keep rows
              listing major cities of that country (e.g., if “USA” is metioned as keyword,
              Keep “New York,” “NY,” “Los Angeles,” “LA,” etc.).
            - For less obvious or ambiguous city references, only keep if you
              are certain they belong to the provided keyword country.
        • Contextual references:
            - If the row strongly implies the filtered concept, keep it.
            - If the connection is extremely tenuous or purely coincidental,
              exclude the row.
        • In cases of doubt, lean towards excluding the row (i.e., do NOT keep
          unless it is fairly certain to be related).
          • For numeric **employee count ranges** (e.g., "11 - 200 employees"):  **Keep rows if the listed employee count range *overlaps or falls within* the user-provided range.**
            - Examples: If user provides "11 - 200", keep "50-200", "100-200", "11-100", "11-500", "150", but exclude "1-10", "300-500", "5".
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
    You are a highly specialized AI filter, designed for conceptual reasoning on spreadsheet data.
    You are now focusing EXCLUSIVELY on the **"{column_name}" column**.
    Understand that spreadsheet columns can contain various types of text or numbers.
    Apply conceptual reasoning to identify matches, considering synonyms, related concepts, and variations typical for **data in spreadsheet columns.**

    **Filter Keywords for "{column_name}"**:
    The user wants to keep rows that conceptually relate to these concepts:
    {', '.join(keywords)}
    """
    if user_context:  # <--- Include User Context in instructions if provided
        system_instructions += f"""
        **IMPORTANT: User Context as a Filter**
         - The user context is the most important factor.
         - If a user context is provided, you must use that as the main criteria to **enhance the conceptual reasoning of the provided keywords** and decide which rows to keep.

        **User Context for "{column_name}" Filtering:**
        {user_context}

        **Reasoning Modifier**:
            {reasoning_text}

        **Inclusion/Exclusion Rules**:
            -  Keep rows that conceptually match the keywords and the user context, unless there is a clear contradiction.
            -  Exclude rows that do not match the keywords and user context, or that have characteristics that are defined as undesirable in the user context.
            - When in doubt, if the text has a plausible link with the context, then keep it.
    """

    else:
      system_instructions += f"""

        **Inclusion/Exclusion Rules**:
            - **only** keep rows that conceptually match the provided keywords.
            - **exclude** rows that do not clearly relate to the provided keywords.
            - In cases of doubt, lean towards excluding the row (i.e., do NOT keep unless it is fairly certain to be related).
      """

    if user_context:
        system_instructions += f"""
    **Your Task (Column-Specific Conceptual Filtering)**:
        For each row, analyze the **"{column_name}" value** and decide:
            1) Does it conceptually relate to *any* of the provided filter concepts for the **"{column_name}" column**, while using the user context as the main rule to follow?
            2) Apply the conceptual reasoning modifier, provided below, to adjust how you interpret the keywords.
            3) Based on your conceptual understanding, label each row as:
            - "KEEP": If the **"{column_name}" value** is conceptually relevant based on the keywords, the user context and the reasoning modifier.
            - "EXCLUDE": Otherwise.
        """
    else:
        system_instructions += f"""
    **Your Task (Column-Specific Conceptual Filtering)**:
            For each row, analyze the **"{column_name}" value** and decide:
             1) Does it conceptually relate to *any* of the provided filter concepts for the **"{column_name}" column**?
              2) Based on your conceptual understanding, label each row as:
                - "KEEP": If the **"{column_name}" value** is conceptually relevant to the keywords.
                - "EXCLUDE": Otherwise.
        """

    summaries_text = "\n".join(row_summaries)

    prompt = f"""
    {system_instructions}

    {reasoning_text}

    Now, analyze the following rows (each has RowIndex, Text) and output your final JSON array:
    {summaries_text}
    """

    return prompt.strip()