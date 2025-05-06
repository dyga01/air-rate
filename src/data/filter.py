"""
This script filters columns from a CSV file and performs additional data cleaning.

Steps:
1. Reads the input CSV file ('unfiltered.csv').
2. Filters the dataframe to retain only specified columns.
3. Drops rows with missing values in the 'review_scores_rating' column.
4. Calculates the length of the 'amenities' column if it exists and removes the original column.
5. Saves the cleaned dataframe to a new CSV file ('filtered_output.csv').

Output:
- A filtered and cleaned CSV file named 'filtered_output.csv' in the current working directory.
"""

import pandas as pd
import ast
import re
import numpy as np

def filter_csv_columns():
    """
    Read a CSV file, remove all columns except the ones specified in the list,
    count the length of amenities column and create a new column called amenities_length,
    remove the original amenities column, and remove rows without review_scores_rating.
    """
    # Hardcoded file names
    input_file = "unfiltered.csv"  # Replace with your actual input file name
    output_file = "filtered_output.csv"  # Replace with your desired output file name

    # Columns to keep
    columns_to_keep = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 'amenities', 'price', 'review_scores_rating']

    try:
        # Read the CSV file
        print(f"Reading CSV file: {input_file}")
        df = pd.read_csv(input_file)

        # Get the original row count
        original_rows = len(df)

        # Get the original column count
        original_columns = df.columns.tolist()
        original_count = len(original_columns)

        # Check which columns from our list exist in the dataframe
        existing_columns = [col for col in columns_to_keep if col in df.columns]

        # Filter to keep only specified columns that exist
        df_filtered = df[existing_columns].copy()

        # Remove rows without review_scores_rating if that column exists
        if 'review_scores_rating' in df_filtered.columns:
            df_filtered = df_filtered.dropna(subset=['review_scores_rating'])

        # Count length of amenities and create new column if amenities column exists
        if 'amenities' in df_filtered.columns:
            # Count length of amenities (assuming it's a string representation of a list)
            df_filtered['amenities_length'] = df_filtered['amenities'].apply(
                lambda x: len(ast.literal_eval(x)) if isinstance(x, str) and pd.notna(x) else 0
            )

            # Remove the original amenities column
            df_filtered = df_filtered.drop('amenities', axis=1)

        # Missing columns
        missing_columns = [col for col in columns_to_keep if col not in df.columns]

        # Save the filtered dataframe
        df_filtered.to_csv(output_file, index=False)

        # Print summary
        print(f"\nFiltering complete:")
        print(f"- Original columns: {original_count}")
        print(f"- Original rows: {original_rows}")
        print(f"- New columns: {df_filtered.shape[1]}")
        print(f"- New rows: {df_filtered.shape[0]}")
        print(f"- Rows removed (no review score): {original_rows - df_filtered.shape[0]}")
        print(f"\nFiltered CSV saved to: {output_file}")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    return True

filter_csv_columns()
