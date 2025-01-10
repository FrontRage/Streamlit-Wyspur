def build_user_instructions(column_keywords_dict: dict) -> str:
    """
    Build a text block that tells the LLM what columns to look at
    and which keywords/concepts to exclude, using AND logic.
    """
    if not column_keywords_dict:
        return "No special exclude instructions provided."
    
    instructions = "Here are the columns and their exclude keywords:\n"

    for col, keywords in column_keywords_dict.items():
        instructions += f"- Column '{col}': exclude if it matches any of ({', '.join(keywords)})\n"

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

        • Exclude rows ONLY if they EXACTLY match the user’s keywords or a well-known,
          direct synonym.
        • Slight rewordings, minor variations, or related (but not identical) terms
          should NOT be excluded.
        • Abbreviations are excluded only if they are definitively the same concept
          (e.g., “CEO” vs. “C.E.O.”).
        • Location matching is applied strictly: exclude ONLY if the column has an
          exact match to the country (or a universally recognized abbreviation).
          For instance, if “USA” is excluded, only exclude rows listing “USA” or
          “U.S.A.”, not a city in the USA unless it explicitly says “USA.”
        • If uncertain whether a term qualifies as an exact match or direct synonym,
          keep the row (do NOT exclude).
        """

    elif slider_value == 5:
        # LEVEL 5: VERY BROAD EXCLUSIONS
        return """
        LEVEL 5/5: VERY BROAD

        • Exclude rows if they even loosely or thematically align with the user’s
          exclude concepts for each column, including synonyms, tangential references, and spelled-out
          or abbreviated forms.
            - Example: If “CEO” is excluded in a column, also exclude “Chief Executive Officer,”
              “C.E.O.,” or “CEOs” (plural).
            - Example: If “VP” is excluded, also exclude “Vice President,”
              “V.P.,” or variations of that title including Senior positions like "SVP"
        • Treat minor variations, rewordings, or partial matches as relevant if they
          are closely related (e.g., “lobbying” for “politics”).
        • Location-based exclusions:
            - If a country is excluded (e.g., “USA,” “US,” “U.S.”), also exclude
              rows that only list the city (e.g., “New York,” “Los Angeles,” etc.).
            - If the user excludes “Germany,” exclude “Berlin,” “Munich,” or any
              German region if identifiable as part of Germany.
        • Err on the side of over-exclusion: if in doubt, exclude the row.
        • Tangential or associated ideas count too: if the user excludes “politics,” in an industry column,
          exclude anything about elections, government agencies, or
          campaign contributions.
        • Partial word overlaps: be mindful of words like “CEOs” (valid) vs. “oceans”
          (not valid). However, if the partial overlap is ambiguous, assume it is
          related and exclude.
        • Company_employeeCountRange column could be a range or just a number, so make sure the provided number
          is within that range for exclusion. 
          Example: key word provided by user "50 or more employees":
          For ranges :"1000_5000" range will fit the exlude criteria. 
          For single number: "1001" in this case, you will exlude as 1001 employees is more than 50.
        """
    else:
        # LEVELS 2, 3, AND 4: MODERATE EXCLUSIONS
        # We treat slider values 2, 3, and 4 as a single "moderate" approach.
        # You can further refine each level (2, 3, and 4) if you want distinct behaviors.
        return f"""
        LEVEL {slider_value}/5: MODERATE APPROACH
        (Note: Levels 2, 3, and 4 belong to this 'moderate' category.)

        • Exclude rows if they match or strongly relate to the user’s exclude concepts.
        • Allow minor or faint references to remain; do NOT exclude unless there is
          a clear, direct, or strong thematic connection.
        • Synonyms and abbreviations:
            - Exclude if the synonym/abbreviation is well-known or commonly used.
            - If it is a subtle or obscure reference, do not exclude.
        • Location handling:
            - If a country (or its standard abbreviation) is excluded, exclude rows
              listing major cities of that country (e.g., if “USA” is excluded,
              exclude “New York,” “NY,” “Los Angeles,” “LA,” etc.).
            - For less obvious or ambiguous city references, only exclude if you
              are certain they belong to the excluded country.
        • Contextual references:
            - If the row strongly implies the excluded concept, exclude it.
            - If the connection is extremely tenuous or purely coincidental,
              keep the row.
        • In cases of doubt, lean towards including the row (i.e., do NOT exclude
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
    Centralize construction of the LLM prompt for a single chunk.
    If debug=False, we omit references to the Reason column to save tokens.
    """
    # Common system instructions
    system_instructions = f"""
    You are an expert data-cleaning assistant. You are going to be provided a Salesforce exported contact list.

    The user wants to exclude a row ONLY if it meets ALL the specified
    column-based keywords (AND-logic). If even one column does not match,
    you must KEEP the row, unless the column that doesnt match doesnt have a value.

    IMPORTANT:
    - Do not include row indices that are not listed in the summaries.
    """

    # If debugging is ON, we request a "Reason" column
    if debug:
        format_instructions = """
        Return pipe-separated (|) text with three columns (no header): 
        RowIndex|Decision|Reason 
        dont add anything else as this structured outpu will be used in a function.

        Where:
        - RowIndex: integer
        - Decision: KEEP or EXCLUDE
        - Reason: an explanation referencing *each column* and 
                  how it matched or did not match the user's exclude keywords.
        """
    else:
        format_instructions = """
        Return pipe-separated (|) text with two columns (no header): 
        RowIndex|Decision
        Dont add anything else as this structured outpu will be used in a function.

        Where:
        - RowIndex: integer
        - Decision: KEEP or EXCLUDE
        """

    prompt_for_llm = f"""
    {system_instructions}

    {user_instructions_text}

    Rows to be analyzed:

    {chr(10).join(row_summaries)}

    This is the reasoning you need to use to exclude columns based on the provided keywords:
    {reasoning_text}

    Now that you have read and understood how broad you need to reason to exclude rows
    based on the column provided keywords go back analyze them and
    {format_instructions}
    """

    return prompt_for_llm.strip()
