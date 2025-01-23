import streamlit as st
import pandas as pd
from io import BytesIO
# Import modules for tools
from modules.fuzzy_logic import python_pre_filter_fuzzy
from utils.prompt_builders import build_user_instructions, build_conceptual_text
from config import MODEL_OPTIONS, DEFAULT_MODEL

# NEW: import your column-by-column approach
from modules.filter_logic import filter_df_master

# ------------------------------
# HELPER FUNCTIONS FOR COMPARISON
# ------------------------------
def read_file(uploaded_file):
    """Read CSV or Excel file into a pandas DataFrame."""
    if uploaded_file.name.lower().endswith('.csv'):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)

def shorten_sheet_name(filename, prefix="Non_Matches_"):
    """
    Creates a safe Excel sheet name, which must be <= 31 characters.
    We'll remove the extension and truncate if necessary.
    """
    base_name = filename.rsplit('.', 1)[0]  # remove extension
    sheet_name = prefix + base_name
    return sheet_name[:31]  # Excel sheet name limit

# ------------------------------
# MAIN TOOLS
# ------------------------------
def compare_tool():
    """
    Compare Files Tool UI logic.
    """
    st.title("Compare Two Files by ProfileId")

    # Let user specify the column name (defaults to 'ProfileId')
    col_name = st.text_input("Enter the column name to match on", "ProfileId")

    st.write("Upload your two CSV or Excel files below:")
    file1 = st.file_uploader("Upload first file", type=["csv", "xlsx", "xls"])
    file2 = st.file_uploader("Upload second file", type=["csv", "xlsx", "xls"])

    if file1 and file2:
        # Read the files into dataframes
        df1 = read_file(file1)
        df2 = read_file(file2)

        # Check that the chosen column exists in both dataframes
        if col_name not in df1.columns or col_name not in df2.columns:
            st.error(f"Column '{col_name}' not found in one or both files. Please check and try again.")
            return

        # Convert the column to string in both dataframes to avoid dtype mismatches
        df1[col_name] = df1[col_name].astype(str)
        df2[col_name] = df2[col_name].astype(str)

        # Find matching IDs
        matching_ids = set(df1[col_name]) & set(df2[col_name])

        # Create DataFrame with Matches: rows in df1 whose ProfileId is also in df2
        df_matches = df1[df1[col_name].isin(matching_ids)].copy()

        # Non-matches in file1
        df_non_matches_file1 = df1[~df1[col_name].isin(matching_ids)].copy()
        # Non-matches in file2
        df_non_matches_file2 = df2[~df2[col_name].isin(matching_ids)].copy()

        # Display small previews
        st.subheader("Preview: Matches (from file1)")
        st.write(df_matches.head())

        st.subheader(f"Preview: Non Matches in {file1.name}")
        st.write(df_non_matches_file1.head())

        st.subheader(f"Preview: Non Matches in {file2.name}")
        st.write(df_non_matches_file2.head())

        # Prepare an Excel file in memory
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Write Matches
            df_matches.to_excel(writer, index=False, sheet_name='Matches')

            # Write Non-matches for File1
            file1_sheet_name = shorten_sheet_name(file1.name)
            df_non_matches_file1.to_excel(writer, index=False, sheet_name=file1_sheet_name)

            # Write Non-matches for File2
            file2_sheet_name = shorten_sheet_name(file2.name)
            df_non_matches_file2.to_excel(writer, index=False, sheet_name=file2_sheet_name)

        # Create a download button
        st.download_button(
            label="Download Comparison Results",
            data=output.getvalue(),
            file_name="comparison_result.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

def calculator_tool():
    """
    Calculator tool UI logic.
    """
    st.header("Calculator")

    num1 = st.number_input("Enter first number", value=0.0, format="%.2f", key="num1_input")
    num2 = st.number_input("Enter second number", value=0.0, format="%.2f", key="num2_input")
    operation = st.selectbox("Select operation", ["Add", "Subtract", "Multiply", "Divide"], key="operation_selectbox")

    if operation == "Add":
        result = num1 + num2
    elif operation == "Subtract":
        result = num1 - num2
    elif operation == "Multiply":
        result = num1 * num2
    elif operation == "Divide":
        result = num1 / num2 if num2 != 0 else "Error: Division by zero"

    st.write(f"Result: {result}")

def filter_tool():
    """
    Filter tool UI logic.
    """
    st.header("Wyspur AI Conceptual CSV Filter")

    # 1) File upload
    uploaded_file = st.file_uploader("Upload a CSV file to filter", type=["csv"])
    if not uploaded_file:
        st.stop()

    # Read CSV
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded CSV")
    st.dataframe(df.head())

    # 2) Column selection & exclude keywords
    all_columns = df.columns.tolist()
    selected_columns = st.multiselect("Pick one or more columns to filter on:", all_columns)

    column_keywords = {}
    for col in selected_columns:
        user_input = st.text_input(f"Enter filter keywords for '{col}' (comma-separated)")
        if user_input.strip():
            keywords_list = [kw.strip() for kw in user_input.split(",") if kw.strip()]
            column_keywords[col] = keywords_list

    # 3) Conceptual reasoning slider & chunk size input
    conceptual_slider = st.slider(
        "Conceptual Reasoning Strictness (1=Very Strict, 5=Very Broad)", 1, 5, 5
    )
    st.write("---")

    conceptual_instructions = build_conceptual_text(conceptual_slider)
    chunk_size = st.number_input(
        "Chunk Size for LLM Processing",
        value=400,
        min_value=1,
        step=1,
        help="Fewer rows per chunk reduces prompt size, but increases the number of LLM calls."
    )

    # 4) Model selection dropdown
    selected_model = st.selectbox(
        "Select the LLM Model:",
        options=list(MODEL_OPTIONS.keys()),
        format_func=lambda key: f"{key} - {MODEL_OPTIONS[key]}",
        index=list(MODEL_OPTIONS.keys()).index(DEFAULT_MODEL),
    )
    st.write(f"**Selected Model:** {selected_model}")

    if selected_model in ["o1", "o1-mini"]:
        st.warning(
            "\u26A0\uFE0F You've selected an expensive model! Consider using 'GPT-4o-mini' for affordability."
        )

    # Debug mode
    debug_mode = st.checkbox("Show LLM debugging info?", value=False)

    # 5) Temperature slider
    temperature = st.slider(
        "Temperature (0=deterministic, 1=creative)",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1
    )

    # 6) Fuzzy pre-filter option
    apply_fuzzy = st.checkbox("Apply Python-based Fuzzy Pre-Filter?", value=False)
    if apply_fuzzy and column_keywords:
        st.info("Using fuzzy pre-filter with threshold=85 to remove obvious matches before LLM filtering.")
        df_pre_filtered, excluded_rows = python_pre_filter_fuzzy(df, column_keywords, threshold=85)
        st.write(f"Rows remaining after fuzzy filter: {len(df_pre_filtered)} / {len(df)}")
        df_for_filtering = df_pre_filtered
    else:
        df_for_filtering = df

    # 7) Build user instructions for LLM (used by OLD approach)
    user_instructions = build_user_instructions(column_keywords)

    # ---------------------------------------------------------------------
    # SESSION STATE CHANGES (to persist the Per-Column results across reruns)
    # ---------------------------------------------------------------------
    # Initialize session_state DataFrames (if they don't exist yet)
    if "filtered_df_col" not in st.session_state:
        st.session_state.filtered_df_col = None
    if "excluded_df_col" not in st.session_state:
        st.session_state.excluded_df_col = None

    # ---------------------
    # NEW: Column-by-Column Approach
    # ---------------------
    if st.button("Filter CSV with LLM (Per-Column Approach)"):
        st.write("Filtering in progress... (Per-Column)")

        # <-- CHANGED: moved filter_df_master call inside this button block
        filtered_df_col = filter_df_master(
            df=df_for_filtering,
            columns_to_check=selected_columns,
            column_keywords=column_keywords,
            chunk_size=chunk_size,
            reasoning_text=conceptual_instructions,
            model=selected_model,
            temperature=temperature,
            debug=debug_mode
        )

        # Determine which rows were excluded by comparing indices
        excluded_df_col = df_for_filtering[~df_for_filtering.index.isin(filtered_df_col.index)]

        # Store in session state for viewing/downloading
        st.session_state.filtered_df_col = filtered_df_col
        st.session_state.excluded_df_col = excluded_df_col

        st.success(
            f"Filtering complete! {len(filtered_df_col)} rows remain "
            f"out of {len(df_for_filtering)} pre-filtered rows."
        )

    # ---------------------------------------------------------------------
    # If we have data in session_state, display the previews & downloads
    # ---------------------------------------------------------------------
    if st.session_state.filtered_df_col is not None:
        st.write("Preview of filtered data (Per-Column approach):")
        st.dataframe(st.session_state.filtered_df_col.head(50))

        st.write("---")
        st.write(f"Number of excluded rows: {len(st.session_state.excluded_df_col)}")
        st.dataframe(st.session_state.excluded_df_col.head(50))

        # Download KEPT rows
        csv_data_filtered = st.session_state.filtered_df_col.to_csv(index=False)
        st.download_button(
            label="Download Filtered (Kept) CSV (Per-Column)",
            data=csv_data_filtered,
            file_name="filtered_output_per_column.csv",
            mime="text/csv",
            key="download_kept_btn"  # give a unique key
        )

        # Download EXCLUDED rows
        csv_data_excluded = st.session_state.excluded_df_col.to_csv(index=False)
        st.download_button(
            label="Download Excluded (Removed) CSV (Per-Column)",
            data=csv_data_excluded,
            file_name="excluded_output_per_column.csv",
            mime="text/csv",
            key="download_excluded_btn"  # give a unique key
        )

# ------------------------------
# AUTHENTICATION + MAIN LAYOUT
# ------------------------------
USERNAME = "test123"
PASSWORD = "test123"

def authenticate(username, password):
    return username == USERNAME and password == PASSWORD

def login_page():
    """
    Display a simple centered login page with a logo and credentials input.
    """
    st.set_page_config(layout="centered")
    placeholder = st.empty()

    with placeholder.container():
        st.image("wyspur.png", width=200)
        st.title("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        login_button = st.button("Login")

        if login_button:
            if authenticate(username, password):
                st.session_state.authenticated = True
                st.session_state.auth_failed = False
            else:
                st.session_state.auth_failed = True

        if st.session_state.get("auth_failed"):
            st.error("Invalid username or password")

def main():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        login_page()
    else:
        # Initialize which page to show
        if "show_filter_tool" not in st.session_state:
            st.session_state.show_filter_tool = True
        if "show_compare_tool" not in st.session_state:
            st.session_state.show_compare_tool = False
        if "show_calculator_tool" not in st.session_state:
            st.session_state.show_calculator_tool = False

        with st.sidebar:
            st.image("wyspur.png", use_container_width=True)
            st.title("Wyspur AI Tools")

            if st.button("AI Filter Tool"):
                st.session_state.show_filter_tool = True
                st.session_state.show_compare_tool = False
                st.session_state.show_calculator_tool = False

            if st.button("Compare Files"):
                st.session_state.show_filter_tool = False
                st.session_state.show_compare_tool = True
                st.session_state.show_calculator_tool = False

            if st.button("Calculator"):
                st.session_state.show_filter_tool = False
                st.session_state.show_compare_tool = False
                st.session_state.show_calculator_tool = True

        # Decide which tool to show:
        if st.session_state.show_filter_tool:
            filter_tool()
        elif st.session_state.show_compare_tool:
            compare_tool()
        elif st.session_state.show_calculator_tool:
            calculator_tool()

if __name__ == "__main__":
    main()
