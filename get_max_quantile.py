import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
file_path = "score_data/max_rrt_score_error.csv"
df = pd.read_csv(file_path)

# Show first few rows (optional)

# Calculate and print the 95th percentile of RRT_Score_error
percentile = df["Max_RRT_Score_Error"].quantile(0.95)
print(f"95th percentile Max_RRT_Score_Error: {percentile:.4f}")

bin_width = 0.01
min_val = df["Max_RRT_Score_Error"].min()
max_val = df["Max_RRT_Score_Error"].max()
bins = np.arange(min_val, max_val + bin_width, bin_width)

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(df["Max_RRT_Score_Error"], bins=bins, edgecolor='black')
plt.axvline(percentile, color='red', linestyle='dashed', linewidth=2, label=f'95th Percentile ({percentile:.2f})')
plt.title("Distribution of Max_RRT_Score_Error")
plt.xlabel("Max_RRT_Score_Error")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()


plt.savefig("score_data/max_quantile.png", dpi=300)