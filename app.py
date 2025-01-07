import streamlit as st
import pandas as pd

# Set page configuration
st.set_page_config(layout="centered")

# Import modules for tools
from modules.filter_logic import filter_df_via_llm_summaries
from modules.fuzzy_logic import python_pre_filter_fuzzy
from utils.prompt_builders import build_user_instructions, build_conceptual_text

# Define models and descriptions for dropdown
from config import MODEL_OPTIONS, DEFAULT_MODEL

# Define credentials
USERNAME = "test123"
PASSWORD = "test123"

# Define color palette
PRIMARY_COLOR = "#F9A12A"  # Orange peel
SECONDARY_COLOR = "#DE4097"  # Hollywood cerise
ACCENT_COLOR = "#5580C1"  # Glaucous
TEXT_COLOR = "#000000"  # Black
BACKGROUND_COLOR = "#FFFFFF"  # White

# Apply custom styles
st.markdown(
    f"""
    <style>
    /* Body styling */
    body {{
        background-color: {BACKGROUND_COLOR};
        color: {TEXT_COLOR};
        font-family: Arial, sans-serif;
    }}

    /* Button styling */
    div.stButton > button {{
        background-color: {PRIMARY_COLOR};
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 16px;
        font-weight: bold;
    }}

    div.stButton > button:hover {{
        background-color: {SECONDARY_COLOR};
    }}

    /* Checkbox styling */
    div[data-baseweb="checkbox"] > label {{
        color: {ACCENT_COLOR};
        font-size: 16px;
    }}

    /* Slider styling */
    div[data-testid="stSlider"] .streamlit-slider {{
        background: linear-gradient(to right, {PRIMARY_COLOR}, {ACCENT_COLOR});
    }}
    div[data-testid="stSlider"] .streamlit-slider > div > div > div {{
        background-color: {SECONDARY_COLOR};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def authenticate(username, password):
    return username == USERNAME and password == PASSWORD


def login_page():
    """
    Display a simple centered login page with a logo and credentials input.
    """
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


def calculator_tool():
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
    st.header("Wyspur AI Conceptual CSV Filter")
    uploaded_file = st.file_uploader("Upload a CSV file to filter", type=["csv"])
    if not uploaded_file:
        st.stop()

    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded CSV")
    st.dataframe(df.head())

    all_columns = df.columns.tolist()
    selected_columns = st.multiselect("Pick one or more columns to filter on:", all_columns)
    column_keywords = {}
    for col in selected_columns:
        user_input = st.text_input(f"Enter exclude concepts for '{col}' (comma-separated)")
        if user_input.strip():
            column_keywords[col] = [kw.strip() for kw in user_input.split(",") if kw.strip()]

    conceptual_slider = st.slider("Conceptual Reasoning Strictness (1=Very Strict, 5=Very Broad)", 1, 5, 5)
    chunk_size = st.number_input("Chunk Size for LLM Processing", value=100, min_value=1, step=1)
    selected_model = st.selectbox("Select the LLM Model:", list(MODEL_OPTIONS.keys()))
    debug_mode = st.checkbox("Show LLM debugging info?", value=True)

    if st.button("Filter CSV with LLM"):
        filtered_df = filter_df_via_llm_summaries(
            df=df,
            user_instructions_text=build_user_instructions(column_keywords),
            columns_to_summarize=selected_columns,
            chunk_size=chunk_size,
            conceptual_slider=conceptual_slider,
            reasoning_text=build_conceptual_text(conceptual_slider),
            model=selected_model,
            temperature=1.0,
            debug=debug_mode
        )
        st.write("Filtering complete!")
        st.dataframe(filtered_df.head(50))
        csv_data = filtered_df.to_csv(index=False)
        st.download_button("Download Filtered CSV", data=csv_data, file_name="filtered_output.csv", mime="text/csv")


def main():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        login_page()
    else:
        with st.sidebar:
            st.image("wyspur.png", use_container_width=True)
            st.title("Wyspur AI Tools")
            if "show_filter_tool" not in st.session_state:
                st.session_state.show_filter_tool = True
            if "show_calculator_tool" not in st.session_state:
                st.session_state.show_calculator_tool = False

            if st.button("AI Filter Tool"):
                st.session_state.show_filter_tool = True
                st.session_state.show_calculator_tool = False

            if st.button("Calculator"):
                st.session_state.show_filter_tool = False
                st.session_state.show_calculator_tool = True

        if st.session_state.show_filter_tool:
            filter_tool()
        elif st.session_state.show_calculator_tool:
            calculator_tool()


if __name__ == "__main__":
    main()
