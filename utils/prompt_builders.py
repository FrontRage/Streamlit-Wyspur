def build_user_instructions(column_keywords_dict: dict) -> str:
    """
    Build a text block that tells the LLM what columns to look at
    and which keywords/concepts to exclude, using AND logic.
    """
    if not column_keywords_dict:
        return "No special exclude instructions provided."

    instructions = (
        "Column matching logic (AND condition):Exclude a row only if it meets all the user’s exclude conditions across the specified columns. If any column does not match, keep the row.\n"
    )

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
        Broad matching (Level 5/5):
        - Consider synonyms, abbreviations, and variations. If the user excludes “CEO,” also exclude “Chief Executive Officer,” “C.E.O.,” “CEOs,” etc. If the user excludes “VP,” also exclude “Vice President,” “V.P.,” “Senior VP,” etc.
        - For employee counts columns, you might be given ranges, as an example if the filter is 50+, ranges like "51_200" or "1001_5000" would qualify as 50+ employees and be excluded.
        - For location-based exclusions: if “USA” or “US” is excluded, exclude rows mentioning “United States,” “New York,” “Los Angeles,” etc. If “Germany” is excluded, exclude rows mentioning “Berlin,” “Munich,” etc.
        - For partial-word ambiguity (e.g., “CEO” vs. “oceans”), keep unless you can reason that they are related within the context of the column tittle.
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
    You are an expert data-cleaning assistant. Decide whether to KEEP or EXCLUDE each row based on the user’s instructions. Follow these rules:

    IMPORTANT:
    - Do not include any row index larger than {max_idx} or smaller than {min_idx}.
    - Do not include row indices that are not listed in the summaries.
    """

    # If debugging is ON, we request a "Reason" column
    if debug:
        format_instructions = """
        Return pipe-separated (|) text with three columns (no header): 
        RowIndex|Decision|Reason

        Where:
        - RowIndex: integer
        - Decision: KEEP or EXCLUDE
        - Reason: a short explanation referencing *each column* and 
                  how it matched or did not match the user's exclude keywords.
        """
    else:
        format_instructions = """
        Return pipe-separated (|) text with two columns (no header): 
        RowIndex|Decision

        Where:
        - RowIndex: integer
        - Decision: KEEP or EXCLUDE
        """

    prompt_for_llm = f"""
    {system_instructions}

    {format_instructions}

    {user_instructions_text}

    {reasoning_text}

    [Row Summaries]
    {chr(10).join(row_summaries)}
    """

    return prompt_for_llm.strip()
