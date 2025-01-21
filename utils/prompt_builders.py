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
        • Use **synonyms, abbreviations**, or spelled-out forms:
        - E.g. "CEO" → "Chief Executive Officer," "CEOs," "C.E.O." "VP" → "Vice President," "Senior VP," "V.P."
        - If "VP of sales" is provided, Keep "SVP of Sales" or variations of that tile, but not a "VP of Product" as this title works in a different department.
        • If the column references **locations** (e.g., "USA"), consider any **cities** or **regions** in that country as relevant.
        • For numeric **ranges** (e.g., "11 - 200 employees"), interpret partial or approximate references:
        - "50-200" or "100-200" or "11-100" fit "11-200." and should be kept.
        - If keyword is 50+, include all the ranges or numbers that are bigger than contain 50 or are bigger, example "1001-5000" would be kept.
        • For partial overlaps in text (e.g., "CEOs" vs. "oceans"), assume it’s relevant if the overlap is plausible; if ambiguous, **EXCLUDE**.
        • If you see *any* conceptual link—synonym, slight rewording,label it "KEEP."
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
          • For numeric **ranges** (e.g., "11 - 200 employees"), interpret partial or approximate references:
        - "50-200" or "100-200" or "11-100" fit "11-200." and should be kept.
        """



def build_llm_prompt_single_col_json(
    row_summaries: list,
    column_name: str,
    keywords: list,
    reasoning_text: str,
) -> str:
    """
    BUILD A JSON-BASED PROMPT FOR ONE COLUMN

    Instructs the LLM to analyze rows for a specific column and return JSON-based decisions:
      - "RowIndex": <integer>
      - "Decision": "KEEP" or "EXCLUDE"
    """
    system_instructions = f"""
    You are an advanced LLM conceptual reasoning filter, focusing on **one column**: "{column_name}".
    Traditional filters cannot catch synonyms, slight variations, ranges, but you can!

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
    """
    
    summaries_text = "\n".join(row_summaries)

    prompt = f"""
    {system_instructions}

    {reasoning_text}

    Now, analyze the following rows (each has RowIndex, Text) and output your final JSON array:
    {summaries_text}
    """

    return prompt.strip()

