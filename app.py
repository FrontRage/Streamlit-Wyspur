import streamlit as st
import pandas as pd

# Import modules for tools
from modules.filter_logic import filter_df_via_llm_summaries
from modules.fuzzy_logic import python_pre_filter_fuzzy
from utils.prompt_builders import build_user_instructions, build_conceptual_text

# Define models and descriptions for dropdown
from config import MODEL_OPTIONS, DEFAULT_MODEL

# Define credentials
USERNAME = "test123"
PASSWORD = "test123"


def authenticate(username, password):
    return username == USERNAME and password == PASSWORD


def login_page():
    """
    Display a simple centered login page with a logo and credentials input.
    """
    # Set the page layout to wide and center elements
    st.set_page_config(layout="centered")

    # Placeholder for the login page content
    placeholder = st.empty()

    # Display login content inside the placeholder
    with placeholder.container():
        # Display the logo
        st.image("wyspur.png", width=200)

        # Login form
        st.title("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.button("Login")
        
        if login_button:
            if authenticate(username, password):
                st.session_state.authenticated = True
            else:
                st.error("Invalid username or password")


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

    # Gather exclude keywords per selected column
    column_keywords = {}
    for col in selected_columns:
        user_input = st.text_input(f"Enter exclude concepts for '{col}' (comma-separated)")
        if user_input.strip():
            keywords_list = [kw.strip() for kw in user_input.split(",") if kw.strip()]
            column_keywords[col] = keywords_list

    # 3) Conceptual reasoning slider & chunk size input
    conceptual_slider = st.slider(
        "Conceptual Reasoning Strictness (1=Very Strict, 5=Very Broad)", 1, 5, 5
    )
    st.write("---")

    # Build conceptual instructions from slider
    conceptual_instructions = build_conceptual_text(conceptual_slider)

    chunk_size = st.number_input(
        "Chunk Size for LLM Processing",
        value=100,
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

    # Show a warning if "o1" or "o1-mini" is selected
    if selected_model in ["o1", "o1-mini"]:
        st.warning(
            "\u26A0\uFE0F You've selected an expensive model! Consider using 'GPT-4o-mini' for affordability."
        )

    # Optional: Debug mode
    debug_mode = st.checkbox("Show LLM debugging info?", value=True)

    # 5) Temperature slider
    temperature = st.slider(
        "Temperature (0=deterministic, 1=creative)",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.1
    )

    # 6) Fuzzy pre-filter option
    apply_fuzzy = st.checkbox("Apply Python-based Fuzzy Pre-Filter?", value=True)
    if apply_fuzzy and column_keywords:
        st.info("Using fuzzy pre-filter with threshold=85 to remove obvious matches before LLM filtering.")
        df_pre_filtered, excluded_rows = python_pre_filter_fuzzy(df, column_keywords, threshold=85)
        st.write(f"Rows remaining after fuzzy filter: {len(df_pre_filtered)} / {len(df)}")
        df_for_filtering = df_pre_filtered
    else:
        df_for_filtering = df

    # 7) Build user instructions for LLM
    user_instructions = build_user_instructions(column_keywords)

    # 8) Filter via LLM
    if st.button("Filter CSV with LLM"):
        st.write("Filtering in progress...")

        filtered_df = filter_df_via_llm_summaries(
            df=df_for_filtering,
            user_instructions_text=user_instructions,
            columns_to_summarize=selected_columns,
            chunk_size=chunk_size,           # Pass the chunk size
            conceptual_slider=conceptual_slider,  # Pass the slider value
            reasoning_text=conceptual_instructions,  # Provide conceptual text
            model=selected_model,           # Pass the selected model
            temperature=temperature,        # Pass the temperature value
            debug=debug_mode
        )

        st.success(
            f"Filtering complete! {len(filtered_df)} rows remain "
            f"out of {len(df_for_filtering)} pre-filtered rows."
        )
        st.write("Preview of filtered data:")
        st.dataframe(filtered_df.head(50))

        # 9) Allow user to download the filtered CSV
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered CSV",
            data=csv_data,
            file_name="filtered_output.csv",
            mime="text/csv"
        )


def main():
    """
    Main function to display the login page or authenticated content.
    """
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        login_page()
    else:
        # Sidebar configuration with company logo
        with st.sidebar:
            st.image("wyspur.png", use_container_width=True)
            st.title("Wyspur AI Tools")

            # Initialize session states
            if "show_filter_tool" not in st.session_state:
                st.session_state.show_filter_tool = True  # Default to showing filter tool
            if "show_calculator_tool" not in st.session_state:
                st.session_state.show_calculator_tool = False

            # Sidebar buttons for navigation
            if st.button("AI Filter Tool"):
                st.session_state.show_filter_tool = True
                st.session_state.show_calculator_tool = False

            if st.button("Calculator"):
                st.session_state.show_filter_tool = False
                st.session_state.show_calculator_tool = True

        # Main content area
        if st.session_state.show_filter_tool:
            filter_tool()
        elif st.session_state.show_calculator_tool:
            calculator_tool()


if __name__ == "__main__":
    main()
