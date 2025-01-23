import streamlit as st
# Set page config for wider layout and custom font (optional)
st.set_page_config(layout="centered", page_title="Smart Filtering App")

import pandas as pd
from io import BytesIO
import json
import datetime
import io

# Import modules for tools
from modules.fuzzy_logic import python_pre_filter_fuzzy
from utils.prompt_builders import build_user_instructions, build_conceptual_text
from config import MODEL_OPTIONS, DEFAULT_MODEL

# Import for column-by-column approach
from modules.filter_logic import filter_df_master

# Supabase imports
from supabase import Client, create_client
import os

# Font styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Code+Pro&display=swap');
body {
    font-family: 'Source Code Pro', monospace;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# SUPABASE AUTHENTICATION
# ------------------------------

def create_supabase_client():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    return create_client(url, key)

supabase = create_supabase_client()

def authenticate_supabase(email, password):
    try:
        response = supabase.auth.sign_in_with_password({"email": email, "password": password})
        return response.user
    except Exception as e:
        st.error(f"Authentication failed: {e}")
        return None

def signup_supabase(email, password):
    try:
        response = supabase.auth.sign_up({"email": email, "password": password})
        st.success("Signup successful! Please check your email to verify.")
        return response
    except Exception as e:
        st.error(f"Signup failed: {e}")
        return None


# ------------------------------
# Helper Functions for Supabase Interactions
# ------------------------------

def save_filter_to_supabase(supabase_client, user_id, filter_name, filter_params_dict):
    """Saves filter parameters to Supabase."""
    try:
        response = supabase_client.table("saved_filters").insert({
            "user_id": user_id,
            "filter_name": filter_name,
            "filter_parameters": json.dumps(filter_params_dict) # Serialize to JSON
        }).execute()
        st.write("Response object from Supabase (Save Filter):") # Keep for debugging
        st.write(response) # Keep for debugging

        st.success(f"Filter '{filter_name}' saved successfully!") # <--- CORRECT INDENTATION
        return True # <--- CORRECT INDENTATION

    except Exception as e:
        st.error(f"An unexpected error occurred while saving filter: {e}")
        return False

def load_saved_filters_from_supabase(supabase_client, user_id):
    """Loads saved filters for a user from Supabase."""
    try:
        response = supabase_client.table("saved_filters").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        return response.data # List of dictionaries  <--- CORRECT INDENTATION - DIRECTLY IN 'try'

    except Exception as e:
        st.error(f"Error loading saved filters: {e}")
        return []

def delete_saved_filter_from_supabase(supabase_client, filter_id):
    """Deletes a saved filter from Supabase."""
    try:
        response = supabase_client.table("saved_filters").delete().eq("id", filter_id).execute()
        if response.error:
            st.error(f"Error deleting saved filter: {response.error.message}")
            return False
        else:
            st.success("Filter deleted successfully.")
            return True
    except Exception as e:
        st.error(f"Error deleting saved filter: {e}")
        return False



def record_filter_history_to_supabase(supabase_client, user_id, filter_params_dict, original_filename, filtered_df):
    """Records filter history to Supabase and uploads filtered CSV to Storage."""
    filtered_csv_url = None # Initialize as None

    try:
        # --- Upload to Supabase Storage ---
        filtered_csv_filename = f"filtered_csv_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        supabase_storage_path = f"filtered_csvs/{user_id}/{filtered_csv_filename}" # Organize by user

        csv_buffer = io.StringIO() # Use StringIO to create a string buffer in memory
        filtered_df.to_csv(csv_buffer, index=False) # Write DataFrame to buffer
        csv_bytes = csv_buffer.getvalue().encode('utf-8') # Get bytes from buffer and encode

        upload_response = supabase_client.storage.from_("filtered-csvs").upload( # Use your bucket name
            path=supabase_storage_path,
            file=csv_bytes,
            file_options={"content-type": "text/csv"} # Set MIME type
        )
        if upload_response.error:
            st.error(f"Error uploading to Supabase Storage: {upload_response.error.message}")
            filtered_csv_url = None # Or handle error appropriately
        else:
            filtered_csv_url = supabase_client.storage.from_("filtered-csvs").get_public_url(supabase_storage_path)
            st.success(f"Filtered CSV uploaded to Supabase Storage: {filtered_csv_url}") # Optional success message for debugging

    except Exception as storage_error:
        st.error(f"Error with Supabase Storage: {storage_error}")
        filtered_csv_url = None


    try:
        response = supabase_client.table("filter_history").insert({
            "user_id": user_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "filter_parameters": json.dumps(filter_params_dict),
            "original_csv_filename": original_filename,
            "filtered_csv_url": filtered_csv_url # Store the URL in the history
        }).execute()
        if response.error:
            st.error(f"Error recording filter history in DB: {response.error.message}")
            return False
        else:
            return True
    except Exception as e:
        st.error(f"Error recording filter history in DB: {e}")
        return False


def load_filter_history_from_supabase(supabase_client, user_id):
    """Loads filter history for a user from Supabase."""
    try:
        response = supabase_client.table("filter_history").select("*").eq("user_id", user_id).order("timestamp", desc=True).execute() # Order by timestamp
        if response.error:
            st.error(f"Error loading filter history: {response.error.message}")
            return []
        else:
            return response.data
    except Exception as e:
        st.error(f"Error loading filter history: {e}")
        return []

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
    """Creates a safe Excel sheet name."""
    base_name = filename.rsplit('.', 1)[0]
    sheet_name = prefix + base_name
    return sheet_name[:31]

# ------------------------------
# MAIN TOOLS
# ------------------------------

def compare_tool():
    """Compare Files Tool UI logic with step-by-step approach."""
    st.title("Compare Two Files by ProfileId")

    if "compare_step" not in st.session_state:
        st.session_state.compare_step = 1

    if st.session_state.compare_step == 1:
        st.session_state.compare_col_name = st.text_input("Enter the column name to match on", "ProfileId")
        st.session_state.compare_file1 = st.file_uploader("Upload first file", type=["csv", "xlsx", "xls"])
        if st.session_state.compare_file1:
            st.session_state.compare_step = 2
            st.rerun()

    elif st.session_state.compare_step == 2:
        st.session_state.compare_file2 = st.file_uploader("Upload second file", type=["csv", "xlsx", "xls"])
        if st.session_state.compare_file2:
            try:
                df1 = read_file(st.session_state.compare_file1)
                df2 = read_file(st.session_state.compare_file2)

                col_name = st.session_state.compare_col_name
                if col_name not in df1.columns or col_name not in df2.columns:
                    st.error(f"Column '{col_name}' not found in one or both files.")
                    st.session_state.compare_step = 1
                    st.rerun()
                    return

                df1[col_name] = df1[col_name].astype(str)
                df2[col_name] = df2[col_name].astype(str)

                matching_ids = set(df1[col_name]) & set(df2[col_name])

                st.session_state.compare_df_matches = df1[df1[col_name].isin(matching_ids)].copy()
                st.session_state.compare_df_non_matches_file1 = df1[~df1[col_name].isin(matching_ids)].copy()
                st.session_state.compare_df_non_matches_file2 = df2[~df2[col_name].isin(matching_ids)].copy()

                st.session_state.compare_step = 3
                st.rerun()

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.session_state.compare_step = 1
                st.rerun()

    elif st.session_state.compare_step == 3:
        if 'compare_df_matches' in st.session_state:
            st.subheader("Preview: Matches (from first file)")
            st.write(st.session_state.compare_df_matches.head())

            st.subheader(f"Preview: Non Matches in {st.session_state.compare_file1.name}")
            st.write(st.session_state.compare_df_non_matches_file1.head())

            st.subheader(f"Preview: Non Matches in {st.session_state.compare_file2.name}")
            st.write(st.session_state.compare_df_non_matches_file2.head())

            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                st.session_state.compare_df_matches.to_excel(writer, index=False, sheet_name='Matches')

                file1_sheet_name = shorten_sheet_name(st.session_state.compare_file1.name)
                st.session_state.compare_df_non_matches_file1.to_excel(writer, index=False, sheet_name=file1_sheet_name)

                file2_sheet_name = shorten_sheet_name(st.session_state.compare_file2.name)
                st.session_state.compare_df_non_matches_file2.to_excel(writer, index=False, sheet_name=file2_sheet_name)

            st.download_button(
                label="Download Comparison Results",
                data=output.getvalue(),
                file_name="comparison_result.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="compare_download_button" # Added key
            )

        if st.button("Back to Upload Files", key="compare_back_to_upload_button"): # Added key
            st.session_state.compare_step = 1
            st.rerun()


def calculator_tool():
    """Calculator tool UI logic with step-by-step approach."""
    st.header("Calculator")

    if "calc_step" not in st.session_state:
        st.session_state.calc_step = 1

    if st.session_state.calc_step == 1:
        st.session_state.calc_num1 = st.number_input("Enter first number", value=0.0, format="%.2f")
        if st.button("Next", key="calc_next_button_step1"): # Added key
            st.session_state.calc_step = 2
            st.rerun()

    elif st.session_state.calc_step == 2:
        st.session_state.calc_num2 = st.number_input("Enter second number", value=0.0, format="%.2f")
        st.session_state.calc_operation = st.selectbox("Select operation", ["Add", "Subtract", "Multiply", "Divide"])
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Back", key="calc_back_button_step2"): # Added key
                st.session_state.calc_step = 1
                st.rerun()
        with col2:
            if st.button("Calculate", key="calc_calculate_button_step2"): # Added key
                st.session_state.calc_step = 3
                st.rerun()

    elif st.session_state.calc_step == 3:
        num1 = st.session_state.calc_num1
        num2 = st.session_state.calc_num2
        operation = st.session_state.calc_operation

        if operation == "Add":
            result = num1 + num2
        elif operation == "Subtract":
            result = num1 - num2
        elif operation == "Multiply":
            result = num1 * num2
        elif operation == "Divide":
            result = num1 / num2 if num2 != 0 else "Error: Division by zero"

        st.write(f"Result: {result}")
        if st.button("Start Over", key="calc_start_over_button_step3"): # Added key
            st.session_state.calc_step = 1
            st.rerun()


def filter_tool():
    """Filter tool UI logic with step-by-step approach."""

    if "filter_step" not in st.session_state:
        st.session_state.filter_step = 1

    if st.session_state.filter_step == 1:
        st.title("Smart filtering powered by WyspurAI")

        st.session_state.filter_uploaded_file = st.file_uploader("Upload a CSV file to filter", type=["csv"])
        if st.session_state.filter_uploaded_file:
            try:
                st.session_state.filter_df = pd.read_csv(st.session_state.filter_uploaded_file)
                st.subheader("Preview of Uploaded CSV")
                st.dataframe(st.session_state.filter_df.head())
                st.session_state.filter_step = 2
                st.rerun()
            except Exception as e:
                st.error(f"Error reading file: {e}")

    elif st.session_state.filter_step == 2:
        st.subheader("Column Selection")
        all_columns = st.session_state.filter_df.columns.tolist()
        st.session_state.filter_selected_columns = st.multiselect("Pick one or more columns to filter on:", all_columns)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Back", key="filter_back_button_step2"): # Added key
                st.session_state.filter_step = 1
                st.rerun()
        with col2:
            if st.button("Next", key="filter_next_button_step2"): # Added key
                if st.session_state.filter_selected_columns:
                    st.session_state.filter_step = 3
                    st.rerun()
                else:
                    st.warning("Please select at least one column.")


    elif st.session_state.filter_step == 3:
        st.subheader("Enter Filter Keywords")
        if "filter_column_keywords" not in st.session_state:
            st.session_state.filter_column_keywords = {}

        for col in st.session_state.filter_selected_columns:
            st.session_state.filter_column_keywords[col] = st.text_input(
                f"Enter filter keywords for '{col}' (comma-separated)",
                value=st.session_state.filter_column_keywords.get(col, "")
            )


        col1, col2 = st.columns(2)
        with col1:
            if st.button("Back", key="filter_back_button_step3"): # Added key
                st.session_state.filter_step = 2
                st.rerun()
        with col2:
            if st.button("Next", key="filter_next_button_step3"): # Added key
                if all(st.session_state.filter_column_keywords.get(col) is not None for col in st.session_state.filter_selected_columns):
                    st.session_state.filter_step = 4
                    st.rerun()
                else:
                    st.warning("Please enter keywords for all selected columns.")




    elif st.session_state.filter_step == 4:
        st.subheader("Conceptual Reasoning & Chunk Size")
        st.session_state.filter_conceptual_slider = st.slider(
            "Conceptual Reasoning Strictness (1=Very Strict, 5=Very Broad)", 1, 5, 5
        )
        st.write("---")

        st.session_state.filter_chunk_size = st.number_input(
            "Chunk Size for LLM Processing",
            value=400,
            min_value=1,
            step=1,
            help="Fewer rows per chunk reduces prompt size, but increases the number of LLM calls."
        )

        st.markdown(
            """
            <style>
                div[data-testid="column"]:nth-of-type(1)
                {
                    border:1px solid red;
                } 

                div[data-testid="column"]:nth-of-type(2)
                {
                    border:1px solid blue;
                    text-align: end;
                } 
            </style>
            """,unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Back", key="filter_back_button_step4"): # Added key
                st.session_state.filter_step = 3
                st.rerun()
        with col2:

            if st.button("Next", key="filter_next_button_step4"): # Added key
                st.session_state.filter_step = 5
                st.rerun()

    elif st.session_state.filter_step == 5:
        st.subheader("Advanced Filtering Options")  # Add this line for better clarity

        st.session_state.filter_temperature = st.slider(
            "Temperature (0=deterministic, 1=creative)",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1
        )

        st.session_state.filter_conceptual_slider = st.slider(
            "Conceptual Reasoning Strictness (1=Very Strict, 5=Very Broad)", 1, 5, 5
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Back", key="filter_back_button_step5"): # Added key
                st.session_state.filter_step = 4
                st.rerun()
        with col2:
            if st.button("Next", key="filter_next_button_step5"): # Added key
                st.session_state.filter_step = 6
                st.rerun()


    elif st.session_state.filter_step == 6:
        st.subheader("Model & Fuzzy Filtering")

        st.session_state.filter_selected_model = st.selectbox(
            "Select the LLM Model:",
            options=list(MODEL_OPTIONS.keys()),
            format_func=lambda key: f"{key} - {MODEL_OPTIONS[key]}",
            index=list(MODEL_OPTIONS.keys()).index(DEFAULT_MODEL),
        )
        if st.session_state.filter_selected_model in ["o1", "o1-mini"]:
            st.warning(
                "\u26A0\uFE0F You've selected an expensive model! Consider using 'GPT-4o-mini' for affordability."
            )

        st.session_state.filter_apply_fuzzy = st.checkbox("Apply Python-based Fuzzy Pre-Filter?", value=False)
        if st.session_state.filter_apply_fuzzy and st.session_state.filter_column_keywords:
            st.info("Using fuzzy pre-filter with threshold=85 to remove obvious matches before LLM filtering.")


        col1, col2 = st.columns(2)
        with col1:
            if st.button("Back", key="filter_back_button_step6"): # Added key
                st.session_state.filter_step = 5
                st.rerun()

        with col2:
            if st.button("Next", key="filter_next_button_step6"): # Added key
                st.session_state.filter_step = 7
                st.rerun()


    elif st.session_state.filter_step == 7:
        st.subheader("Confirm and Filter")
        st.write("Review your settings and click 'Start Filtering' to proceed.")
        st.session_state.filter_debug_mode = st.checkbox("Show LLM debugging info?", value=False)

        with st.expander("Save Filter (Optional)"): # Add expander for saving filters
                save_filter_name = st.text_input("Save filter as:", key="save_filter_name_input")
                if st.button("Save Filter", key="save_filter_button_step7"): # Added key
                    if save_filter_name:
                        current_user = st.session_state.user # User is already in session state
                        if current_user:
                            filter_params = { # Capture current filter parameters
                                "selected_columns": st.session_state.filter_selected_columns,
                                "column_keywords": st.session_state.filter_column_keywords,
                                "conceptual_slider": st.session_state.filter_conceptual_slider,
                                "chunk_size": st.session_state.filter_chunk_size,
                                "temperature": st.session_state.filter_temperature,
                                "selected_model": st.session_state.filter_selected_model,
                                "apply_fuzzy": st.session_state.filter_apply_fuzzy,
                                # Add any other relevant filter parameters you want to save
                            }
                            save_filter_to_supabase(supabase, current_user.id, save_filter_name, filter_params)
                        else:
                            st.warning("You must be logged in to save filters.")
                    else:
                        st.warning("Please enter a name for your filter to save.")

        if st.button("Start Filtering", key="filter_start_filtering_button_step7"): # Added key
            st.write("Filtering in progress...")
            conceptual_instructions = build_conceptual_text(st.session_state.filter_conceptual_slider)

            df_for_filtering = st.session_state.filter_df.copy()
            if st.session_state.filter_apply_fuzzy and st.session_state.filter_column_keywords:
                df_pre_filtered, excluded_rows = python_pre_filter_fuzzy(df_for_filtering, st.session_state.filter_column_keywords, threshold=85)
                st.write(f"Rows remaining after fuzzy filter: {len(df_pre_filtered)} / {len(df_for_filtering)}")
                df_for_filtering = df_pre_filtered


            filtered_df_col = filter_df_master(
                df=df_for_filtering,
                columns_to_check=st.session_state.filter_selected_columns,
                column_keywords={k: [v] if isinstance(v, str) else v.split(',') for k, v in st.session_state.filter_column_keywords.items()},
                chunk_size=st.session_state.filter_chunk_size,
                reasoning_text=conceptual_instructions,
                model=st.session_state.filter_selected_model,
                temperature=st.session_state.filter_temperature,
                debug=st.session_state.filter_debug_mode
            )

            st.session_state.filter_filtered_df_col = filtered_df_col
            st.session_state.filter_excluded_df_col = df_for_filtering[~df_for_filtering.index.isin(filtered_df_col.index)]

            # --- RECORD FILTER HISTORY ---
            current_user = st.session_state.user
            if current_user:
                filter_params_history = { # Capture filter parameters for history
                    "selected_columns": st.session_state.filter_selected_columns,
                    "column_keywords": st.session_state.filter_column_keywords,
                    "conceptual_slider": st.session_state.filter_conceptual_slider,
                    "chunk_size": st.session_state.filter_chunk_size,
                    "temperature": st.session_state.filter_temperature,
                    "selected_model": st.session_state.filter_selected_model,
                    "apply_fuzzy": st.session_state.filter_apply_fuzzy,
                }
                original_filename = st.session_state.filter_uploaded_file.name # Get original filename
                record_filter_history_to_supabase(supabase, current_user.id, filter_params_history, original_filename, st.session_state.filter_filtered_df_col)

                st.session_state.filter_step = 8
                st.rerun()

        col1, col2 = st.columns(2)
        with col1:

            if st.button("Back", key="filter_back_button_step7"): # Added key
                st.session_state.filter_step = 6
                st.rerun()



    elif st.session_state.filter_step == 8:
        st.subheader("Filtering Results")
        if 'filter_filtered_df_col' in st.session_state:
            st.write("Preview of filtered data (Per-Column approach):")
            st.dataframe(st.session_state.filter_filtered_df_col.head(50))

            st.write("---")
            st.write(f"Number of excluded rows: {len(st.session_state.filter_excluded_df_col)}")
            st.dataframe(st.session_state.filter_excluded_df_col.head(50))

            # Download KEPT rows
            csv_data_filtered = st.session_state.filter_filtered_df_col.to_csv(index=False)
            st.download_button(
                label="Download Filtered (Kept) CSV",
                data=csv_data_filtered,
                file_name="filtered_output_per_column.csv",
                mime="text/csv",
                key="download_kept_btn"
            )

            # Download EXCLUDED rows
            csv_data_excluded = st.session_state.filter_excluded_df_col.to_csv(index=False)
            st.download_button(
                label="Download Excluded (Removed) CSV",
                data=csv_data_excluded,
                file_name="excluded_output_per_column.csv",
                mime="text/csv",
                key="download_excluded_btn"
            )
        if st.button("Start Over", key="filter_start_over_button_step8"): # Added key
            st.session_state.filter_step = 1
            st.rerun()



# ------------------------------
# AUTHENTICATION + MAIN LAYOUT
# ------------------------------


def login_signup_page():  # Make sure this function is defined
    placeholder = st.empty()

    with placeholder.container():
        st.image("wyspur.png", width=200)
        st.title("Login/Sign Up")

        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        login_col, signup_col = st.columns(2)
        with login_col:
            login_button = st.button("Login", key="login_button_login_page") # Added key
        with signup_col:
            signup_button = st.button("Sign Up", key="signup_button_login_page") # Added key

        if login_button:
            user = authenticate_supabase(email, password)
            if user:
                st.session_state.authenticated = True
                st.session_state.user = user
                st.session_state.auth_failed = False
                st.rerun()
        elif signup_button:
            response = signup_supabase(email, password)
            if response:
                st.success("Signup successful! Please check your email to verify your account.")

        if st.session_state.get("auth_failed"):
            st.error("Invalid email or password")


def main():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.auth_failed = False

    if not st.session_state.authenticated:
        login_signup_page()
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

            if st.button("AI Filter Tool", key="sidebar_filter_tool_button"): # Added key
                st.session_state.show_filter_tool = True
                st.session_state.show_compare_tool = False
                st.session_state.show_calculator_tool = False
                st.session_state.pop('filter_step', None) # Reset filter steps

            if st.button("Compare Files", key="sidebar_compare_files_button"): # Added key
                st.session_state.show_filter_tool = False
                st.session_state.show_compare_tool = True
                st.session_state.show_calculator_tool = False
                st.session_state.pop('compare_step', None) # Reset compare steps

            if st.button("Calculator", key="sidebar_calculator_button"): # Added key
                st.session_state.show_filter_tool = False
                st.session_state.show_compare_tool = False
                st.session_state.show_calculator_tool = True
                st.session_state.pop('calc_step', None) # Reset calculator steps

            # Display user info and logout button (only when logged in)
            if 'user' in st.session_state: # Check if user exists in session_state
                user = st.session_state.user
                st.sidebar.write(f"Welcome, {user.email}")
            if st.sidebar.button("Logout", key="sidebar_logout_button"): # Added key
                supabase.auth.sign_out()
                del st.session_state.authenticated
                if 'user' in st.session_state: # Safely delete user from session_state
                    del st.session_state.user
                st.rerun()

        # Tabs for different functionalities
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["CSV Filter", "Compare Files", "Calculator", "Saved Filters", "Filter History"]) # Add "Saved Filters" tab

        with tab1: # "CSV Filter" tab
            filter_tool()
        with tab2: # "Compare Files" tab
            compare_tool()
        with tab3: # "Calculator" tab
            calculator_tool()
        with tab4: # "Saved Filters" tab
            saved_filters_tab_content() # Function to handle saved filters tab content
        with tab5: # "Filter History" tab
            filter_history_tab_content() # Function for history tab content

def filter_history_tab_content():
    """Content for the Filter History tab."""
    st.header("Filter History")
    current_user = st.session_state.user
    if current_user:
        user_id = current_user.id
        filter_history_entries = load_filter_history_from_supabase(supabase, user_id)

        if filter_history_entries:
            for history_item in filter_history_entries:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(f"**{history_item['timestamp']}**")
                    st.write(f"Original File: `{history_item['original_csv_filename']}`")
                    with st.expander("Filter Parameters"):
                        st.json(json.loads(history_item['filter_parameters'])) # Display parameters
                with col2:
                    st.warning("Download not yet implemented", icon="⚠️") # Placeholder for download button later
                    # In the next phase, we'll add download button here if we implement CSV storage

        else:
            st.info("No filter history yet.")
    else:
        st.info("Log in to see your filter history.")

def saved_filters_tab_content():
    """Content for the Saved Filters tab."""
    st.header("Saved Filters")
    current_user = st.session_state.user
    if current_user:
        user_id = current_user.id
        saved_filters = load_saved_filters_from_supabase(supabase, user_id)

        if saved_filters:
            for filter_data in saved_filters:
                col1, col2, col3 = st.columns([3, 1, 1]) # Example layout for each saved filter
                with col1:
                    st.subheader(filter_data['filter_name'])
                    # Optionally display some filter parameters summary here if needed
                with col2:
                    if st.button("Apply Filter", key=f"apply_filter_button_{filter_data['id']}"): # Added key - unique per filter
                        # Load filter_data['filter_parameters'] and populate UI
                        filter_params = json.loads(filter_data['filter_parameters'])
                        apply_saved_filter_to_ui(filter_params) # Function to apply to UI (see next step)
                        st.switch_to_tab("CSV Filter") # Switch to the main filter tab (adjust tab name if needed)
                with col3:
                    if st.button("Delete", key=f"delete_filter_button_{filter_data['id']}"): # Added key - unique per filter
                        if delete_saved_filter_from_supabase(supabase, filter_data['id']):
                            st.rerun() # Refresh to update the list
        else:
            st.info("No saved filters yet.")
    else:
        st.info("Log in to see your saved filters.")

def apply_saved_filter_to_ui(filter_params):
    """Applies saved filter parameters to the filter tool UI."""
    # Set session state variables to populate the filter tool UI
    st.session_state.filter_selected_columns = filter_params.get("selected_columns", [])
    st.session_state.filter_column_keywords = filter_params.get("column_keywords", {})
    st.session_state.filter_conceptual_slider = filter_params.get("conceptual_slider", 5) # Default if not in saved filter
    st.session_state.filter_chunk_size = filter_params.get("chunk_size", 400) # Default if not in saved filter
    st.session_state.filter_temperature = filter_params.get("temperature", 0.0) # Default if not in saved filter
    st.session_state.filter_selected_model = filter_params.get("selected_model", DEFAULT_MODEL) # Default if not in saved filter
    st.session_state.filter_apply_fuzzy = filter_params.get("apply_fuzzy", False) # Default if not in saved filter
    st.session_state.filter_step = 3 # Go to the "Enter Filter Keywords" step after applying a saved filter

    # You might need to also set the text input values for keywords.
    # Since the UI is built step by step, setting session state should trigger UI updates on rerun.
    st.session_state.show_filter_tool = True # Ensure filter tool is shown if not already   



    # Decide which tool to show: - Removed redundant tool selection and sidebar/logout buttons from here
    # These should only be in the main() function to avoid duplication and potential conflicts.


if __name__ == "__main__":
    main()