"""
This script combines multiple CSV files located in the 'locationCSV' directory into a single CSV file.

Steps:
1. Reads all CSV files in the 'locationCSV' directory.
2. Appends each file's data into a list of dataframes.
3. Concatenates all dataframes into a single dataframe.
4. Writes the combined dataframe to a new CSV file named 'combined_data.csv'.

Output:
- A single CSV file named 'combined_data.csv' in the current working directory.
"""

import pandas as pd
import glob
import os

# Path to your folder containing CSV files
local_path = os.getcwd()
file_path = os.path.join(local_path, 'locationCSV')

# Get all CSV files in the folder
all_files = glob.glob(os.path.join(file_path, "*.csv"))

# List to store individual dataframes
dfs = []

# Read and append each CSV file
for filename in all_files:
    df = pd.read_csv(filename)
    dfs.append(df)

# Combine all dataframes
combined_df = pd.concat(dfs, ignore_index=True)

# Write to a new CSV file
combined_df.to_csv(os.path.join(local_path, 'combined_data.csv'), index=False)

print(f"Combined {len(all_files)} CSV files into 'combined_data.csv'")
