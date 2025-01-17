import streamlit as st
import pandas as pd
from io import BytesIO

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

def main():
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

        # Display small previews (optional)
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

if __name__ == "__main__":
    main()
