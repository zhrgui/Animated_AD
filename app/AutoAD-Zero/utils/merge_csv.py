import pandas as pd
import glob
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_directory', required=True, type=str, help='Directory containing CSV files to merge')
    parser.add_argument('--output_file', required=True, type=str, help='Path for the merged output CSV')
    args = parser.parse_args()

    # Use glob to find all CSV files in the directory
    csv_files = sorted(glob.glob(os.path.join(args.csv_directory, "*")))

    # Read and concatenate all CSV files
    df_list = [pd.read_csv(file) for file in csv_files]
    merged_df = pd.concat(df_list, ignore_index=True)

    # Save the merged dataframe to a new CSV file
    merged_df.to_csv(args.output_file, index=False)

    print(f"Merged CSV saved to {args.output_file}")
