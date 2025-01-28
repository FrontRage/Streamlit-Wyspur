import streamlit as st
from streamlit_lottie import st_lottie
# Set page config for wider layout and custom font (optional)
import streamlit.components.v1 as components
st.set_page_config(layout="centered", page_title="wysper Ai")

import pandas as pd
from io import BytesIO
import json
import datetime
import io
import time

# Import modules for tools
from modules.fuzzy_logic import python_pre_filter_fuzzy
from utils.prompt_builders import build_user_instructions, build_conceptual_text
from config import MODEL_OPTIONS, DEFAULT_MODEL

# Import for column-by-column approach
from modules.filter_logic import filter_df_master

# Supabase imports
from supabase import Client, create_client
import os

# ------------------------------
# Streamlit CSS custimization
# ------------------------------


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
        # Removed incorrect if response.error check
        st.success("Filter deleted successfully.") # Success message now directly in try block
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
        # Removed incorrect if response.error check - no success message here as it's a background task
        return True
    except Exception as e:
        st.error(f"Error recording filter history in DB: {e}")
        return False


def load_filter_history_from_supabase(supabase_client, user_id):
    """Loads filter history for a user from Supabase."""
    try:
        response = supabase_client.table("filter_history").select("*").eq("user_id", user_id).order("timestamp", desc=True).execute() # Order by timestamp
        # Removed incorrect if response.error check
        return response.data
    except Exception as e:
        st.error(f"Error loading filter history: {e}")
        return []

# ------------------------------
# HELPER FUNCTIONS FOR Lottie Animation
# ------------------------------

def stream_greeting_message(username): # Function to stream greeting text WITH <br> for line breaks
    greeting = f"""Hello, {username}!<br>"""  # Use <br> for line break after "Hello, username!"
    greeting += """Ready to unleash the power of AI to filter your CSV data?<br>""" # <br> after the question
    greeting += """Let's get started!🚀<br>""" # <br> after "Let's get started!"
    greeting += """Step 1. Upload your CSV file below.""" # No <br> at the very end if you don't want extra space

    for word in greeting.split(" "):
        yield word + " "
        time.sleep(0.07) # Adjust speed as needed

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
# Streamlit Custom Components width: 100%;  /* Optional: Make buttons full width */
# ------------------------------

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# ------------------------------
# MAIN TOOLS
# ------------------------------



def filter_tool():

    """Filter tool UI logic with step-by-step approach."""


    if "filter_step" not in st.session_state:
        st.session_state.filter_step = 1
    if "step1_initialized" not in st.session_state: # Initialize session state for step 1 load tracking
        st.session_state.step1_initialized = False

    if st.session_state.filter_step == 1:

        step1_col1, step1_col2, step1_col3 = st.columns([1, 3, 1]) # Create 3 columns

        with step1_col1:
            # Inject CSS to left content in the left column (for animation)
            st.markdown(
                    """
                    <style>
                        .stColumn:nth-child(2) { /* Target the second stColumn within stHorizontalBlock (adjust index if needed) */
                            background-color: #e6f7ff; /* Light blue bubble background */
                            border: 2px solid #91bfdb; /* Blue border */
                            border-radius: 15px;
                            padding: 15px;
                            margin-bottom: 10px;
                            text-align: left; /* Keep text alignment left within the bubble */
                            display: inline-block; /* To make bubble wrap content */
                            vertical-align: top; /* Align bubble to the top of the column */
                        }
                        .stColumn:nth-child(3) p { /* Style paragraph text INSIDE the bubble */
                            margin: 0;
                            word-wrap: break-word;
                        }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
            # Middle column - Lottie Animation
            lottie_filepath = "lottie/wyspur_lottie.json" # Replace with your file path
            lottie_json = load_lottiefile(lottie_filepath)

            if lottie_json:
                st_lottie(
                    lottie_json,
                    speed=0.1,
                    reverse=False,
                    loop=True,
                    quality="high",
                    height=300 / 2,
                    width=300 / 2,
                    key=None,
                )
            else:
                st.error(f"Failed to load Lottie animation from file: {lottie_filepath}")
        

        with step1_col2:
            # Middle column - Greeting Text (Chat Bubble Style - Font Change AND Error Fix - CORRECTED)
            user = st.session_state.user

            if not st.session_state.step1_initialized:
                time.sleep(0.0)

                # Inject CSS for chat bubble using st.markdown (targeting column 2 now) - FONT CHANGE and other styles
                st.markdown(
                    """
                    <style>
                        .stColumn:nth-child(2) { /* Target the SECOND stColumn within stHorizontalBlock */
                            background-color: #e6f7ff;
                            border: 2px solid #91bfdb;
                            border-radius: 15px;
                            padding: 8px; /* Reduced padding */
                            margin-bottom: 10px;
                            text-align: left;
                            display: inline-block;
                            vertical-align: top;
                        }
                        .chat-bubble-text-js { /* Class for text INSIDE the bubble - FONT CHANGED */
                            margin: 0;
                            word-wrap: break-word;
                            white-space: pre-line;
                            font-family: 'Courier New', monospace; /* Font changed to Courier New, monospace */
                        }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                greeting_text_full = stream_greeting_message(user.email.split('@')[0]) # Get greeting text from Python function

                # Use st.components.v1.html to inject chat bubble and JavaScript streaming (in column 2)
                components.html(
                    f"""
                    <div class="chat-bubble">
                        <p class="chat-bubble-text-js" id="chat-bubble-text-area"></p>  <!-- Target for JS streaming -->
                    </div>

                    <script>
                        function streamText(text, elementId) {{
                            let words = text.split(' ');
                            let index = 0;
                            let intervalId = setInterval(() => {{
                                if (index < words.length) {{
                                    document.getElementById(elementId).innerHTML += words[index] + ' ';
                                    index++;
                                }} else {{
                                    clearInterval(intervalId);
                                }}
                            }}, 70); // Adjust speed as needed
                        }}

                        let greeting = `{ "".join(stream_greeting_message(user.email.split('@')[0])) }`; // <-----  CRITICAL LINE: Pass the FUNCTION'S OUTPUT as a STRING
                        streamText(greeting, 'chat-bubble-text-area'); // Call JS streaming function
                    </script>
                    """,
                    height=100, # Adjust height as needed
                    scrolling=False,
                )


                st.session_state.step1_initialized = True # Mark step 1 as initialized
            
      

        with step1_col3:
            # Right most column - Spacer (can be empty or minimal content)
            pass  

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
        if 'filter_df' in st.session_state:  # Check if filter_df exists in session state
            all_columns = st.session_state.filter_df.columns.tolist()

            # --- THE FIX IS HERE ---
            default_columns = st.session_state.get("filter_selected_columns", [])
            selected_columns = st.multiselect(
                "Pick one or more columns to filter on:",
                all_columns,
                default=default_columns,  # Use 'default' to pre-select
                key="filter_multiselect_step2"  # Important: Add a key here!
            )
            st.session_state.filter_selected_columns = selected_columns # Update the session state

            # --- ---------------- ---

            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                if st.button("Back", key="filter_back_button_step2"):
                    st.session_state.filter_step = 1
                    st.rerun()
            with col2:
                if st.button("Reset Filter", key="filter_reset_button_step2"): # Reset button
                    reset_filter_parameters()
                    st.session_state.filter_step = 1  # Go back to step 1 after reset. Change if different reset behaviour is needed
                    st.rerun()
            with col3:
                if st.button("Next", key="filter_next_button_step2"):
                    if st.session_state.filter_selected_columns:
                        st.session_state.filter_step = 3
                        st.rerun()
                    else:
                        st.warning("Please select at least one column.")

        else:
            st.warning("Please upload a CSV file first.") # Handle case where no file is uploaded yetlect at least one column.")


    elif st.session_state.filter_step == 3:
        st.subheader("Enter Filter Keywords")
        if "filter_column_keywords" not in st.session_state:
            st.session_state.filter_column_keywords = {}

        for col in st.session_state.filter_selected_columns:
            st.session_state.filter_column_keywords[col] = st.text_input(
                f"Enter filter keywords for '{col}' (comma-separated)",
                value=st.session_state.filter_column_keywords.get(col, "")
            )


        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            if st.button("Back", key="filter_back_button_step3"): # Added key
                st.session_state.filter_step = 2
                st.rerun()
        with col2:
            if st.button("Reset Filter", key="filter_reset_button_step3"): # Reset button
                reset_filter_parameters()
                st.session_state.filter_step = 1  # Go back to step 1 after reset. Change if different reset behaviour is needed
                st.rerun()
        with col3:
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


        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            if st.button("Back", key="filter_back_button_step4"): # Added key
                st.session_state.filter_step = 3
                st.rerun()
        with col2:
            if st.button("Reset Filter", key="filter_reset_button_step4"): # Reset button
                reset_filter_parameters()
                st.session_state.filter_step = 1  # Go back to step 1 after reset. Change if different reset behaviour is needed
                st.rerun()
        with col3:

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

        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            if st.button("Back", key="filter_back_button_step5"): # Added key
                st.session_state.filter_step = 4
                st.rerun()
        with col2:
            if st.button("Reset Filter", key="filter_reset_button_step5"): # Reset button
                reset_filter_parameters()
                st.session_state.filter_step = 1  # Go back to step 1 after reset. Change if different reset behaviour is needed
                st.rerun()
        with col3:
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


        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            if st.button("Back", key="filter_back_button_step6"): # Added key
                st.session_state.filter_step = 5
                st.rerun()
        with col2:
            if st.button("Reset Filter", key="filter_reset_button_step6"): # Reset button
                reset_filter_parameters()
                st.session_state.filter_step = 1  # Go back to step 1 after reset. Change if different reset behaviour is needed
                st.rerun()
        with col3:
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
                filter_column_config=st.session_state.filter_column_config, # <--- DIRECTLY PASS filter_column_config
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

        col1, col2= st.columns(2)
        with col1:

            if st.button("Back", key="filter_back_button_step7"): # Added key
                st.session_state.filter_step = 6
                st.rerun()
        with col2:
            if st.button("Reset Filter", key="filter_reset_button_step7"): # Reset button
                reset_filter_parameters()
                st.session_state.filter_step = 1  # Go back to step 1 after reset. Change if different reset behaviour is needed
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
            st.session_state.step1_initialized = False # Reset step 1 initialization flag
            st.rerun()



# ------------------------------
# AUTHENTICATION + MAIN LAYOUT
# ------------------------------


import streamlit as st

def login_signup_page():
    # --- Example: minimal custom CSS for centering ---
    st.markdown(
        """
        <style>
         /* --- Login/Signup Button Styling - Kept from Iteration 4/5 (Pinker Colors & Size) --- */
        div.stButton > button:first-child,
        div.stButton > button:nth-of-type(2) {
            background: linear-gradient(to right, #F770B7, #D546A2); /* Pinker Wyspur Gradient */
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.5em 1.2em;
            font-weight: bold;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
            transition: background-color 0.3s ease;
            display: inline-block;
            width: auto;
            min-width: 85px;
            height: auto;
            min-height: 36px;
            box-sizing: border-box;
            text-align: center;
            font-size: 0.9rem;
        }

        div.stButton > button:first-child:hover,
        div.stButton > button:nth-of-type(2):hover {
            background: linear-gradient(to right, #FF99CC, #F066B3); /* Lighter pink hover gradient */
        }

        /* Center almost everything inside a 'centered' container */
        .centered {
            display: flex;
            flex-direction: column;
            align-items: center;  /* horizontal centering */
            justify-content: center;  /* vertical centering if needed */
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Create an overall empty container so we can control layout
    placeholder = st.empty()
    with placeholder.container():

        # Use three columns and drop the content into the middle column
        # so that the logo and title appear centered.
        top_col1, top_col2, top_col3 = st.columns([1,2,1])
        with top_col2:
            st.markdown("<div class='centered'>", unsafe_allow_html=True)
            st.image("wyspur.png", width=400)
            st.title("Account Login")
            st.markdown("</div>", unsafe_allow_html=True)
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
        # If you also want the text inputs centered, just reuse a centered block


        # Now split the bottom portion into columns so that
        # Login is on the left and Sign Up is on the right.
        bottom_col1, _, bottom_col2 = st.columns([1,2,1])  # middle column is just spacer
        with _:
            login_button = st.button("Login", key="login_button_login_page")
        #with bottom_col2:
            #signup_button = st.button("Sign Up", key="signup_button_signup_page")

        # Then handle authentication logic, etc.
        if login_button:
            user = authenticate_supabase(email, password)
            if user:
                st.session_state.authenticated = True
                st.session_state.user = user
                st.session_state.auth_failed = False
                st.rerun()

        #elif signup_button:
        #    response = signup_supabase(email, password)
        #    if response:
        #        st.success("Signup successful! Check your email to verify your account.")

        if st.session_state.get("auth_failed"):
            st.error("Invalid email or password")
fake_display = st.empty



def main():

    st.markdown(
        """
        <style>
         /* --- Login/Signup Button Styling - Kept from Iteration 4/5 (Pinker Colors & Size) --- */
        div.stButton > button:first-child,
        div.stButton > button:nth-of-type(2) {
            background: linear-gradient(to right, #F770B7, #D546A2); /* Pinker Wyspur Gradient */
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.5em 1.2em;
            font-weight: bold;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
            transition: background-color 0.3s ease;
            display: inline-block;
            width: auto;
            min-width: 85px;
            height: auto;
            min-height: 36px;
            box-sizing: border-box;
            text-align: center;
            font-size: 0.9rem;
        }

        div.stButton > button:first-child:hover,
        div.stButton > button:nth-of-type(2):hover {
            background: linear-gradient(to right, #FF99CC, #F066B3); /* Lighter pink hover gradient */
        }

        /* Center almost everything inside a 'centered' container */
        .centered {
            display: flex;
            flex-direction: column;
            align-items: center;  /* horizontal centering */
            justify-content: center;  /* vertical centering if needed */
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.auth_failed = False

    if not st.session_state.authenticated:
        login_signup_page()
    else:


        # Initialize which page to show
        if "show_filter_tool" not in st.session_state:
            st.session_state.show_filter_tool = True


        # Inject CSS to center sidebar content
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"] {
                text-align: center;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        with st.sidebar:
            st.image("wyspur.png", use_container_width=True)
            st.title("AI Tools")

            if st.button("AI CSV FILTER ", key="sidebar_filter_tool_button"): # Added key
                st.session_state.show_filter_tool = True
                st.session_state.pop('filter_step', None) # Reset filter steps
                st.session_state.step1_initialized = False # Reset step 1 initialization when going back to filter tool


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

        tab_names = ["CSV Filter", "Saved Filters", "Filter History"]
        tab1, tab2, tab3 = st.tabs(tab_names)

        with tab1:
            filter_tool()
        with tab2:
            saved_filters_tab_content()
        with tab3:
            filter_history_tab_content()


def filter_history_tab_content():
    """Content for the Filter History tab."""
    st.header("Filtered CSV History")
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
    st.header("Filter Name")
    current_user = st.session_state.user
    if current_user:
        user_id = current_user.id
        saved_filters = load_saved_filters_from_supabase(supabase, user_id)

        if saved_filters:
            for filter_data in saved_filters:
                col1, col2, col3 = st.columns([3, 1, 1]) # Example layout for each saved filter
                with col1:
                    st.write(filter_data['filter_name'])
                    # Optionally display some filter parameters summary here if needed
                with col2:
                    if st.button("Apply Filter", key=f"apply_filter_button_{filter_data['id']}"): # Added key - unique per filter
                        # Load filter_data['filter_parameters'] and populate UI
                        filter_params = json.loads(filter_data['filter_parameters'])
                        apply_saved_filter_to_ui(filter_params) # Function to apply to UI (see next step)

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
    st.info("Saved filter parameters applied! Go to the 'CSV Filter' tab to use them.") # ADD SUCCESS MESSAGE

def reset_filter_parameters():
    """Resets all filter parameters in session state."""
    st.session_state.filter_selected_columns = []
    st.session_state.filter_column_keywords = {}
    st.session_state.filter_conceptual_slider = 5  # Reset to default value
    st.session_state.filter_chunk_size = 400  # Reset to default value
    st.session_state.filter_temperature = 0.0  # Reset to default value
    st.session_state.filter_selected_model = DEFAULT_MODEL  # Reset to default model
    st.session_state.filter_apply_fuzzy = False  # Reset to default value
    st.session_state.step1_initialized = False # Reset step 1 initialization flag
    # Reset any other filter-related session state variables

    # If you're using the saved_filter_applied flag, reset it as well:
    st.session_state.saved_filter_applied = False

    st.info("Filter parameters reset!") # Confirmation message



if __name__ == "__main__":
    main()