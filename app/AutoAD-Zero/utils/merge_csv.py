import pandas as pd
import glob
import os

# Set the directory containing your CSV files
csv_directory = "/scratch/shared/beegfs/zhongrui/autoad_data/results/0330_autoad/tgrp_ttf_dinov2_0.6/stage2_results"  # Update this path

# Use glob to find all CSV files in the directory
csv_files = sorted(glob.glob(os.path.join(csv_directory, "*")))

# Read and concatenate all CSV files
df_list = [pd.read_csv(file) for file in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)

# Save the merged dataframe to a new CSV file
merged_df.to_csv("/scratch/shared/beegfs/zhongrui/autoad_data/results/0330_autoad/tgrp_ttf_dinov2_0.6/stage2_results.csv", index=False)

print("Merged CSV saved as merged_output.csv")