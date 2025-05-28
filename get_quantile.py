import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
file_path = "score_data/rrt_score_log.csv"
df = pd.read_csv(file_path)


# Calculate the 95th percentile of RRT_Score_error
percentile = df["RRT_Score_error"].quantile(0.95)
print(f"95th percentile RRT_Score_error: {percentile:.4f}")


bin_width = 0.01
min_val = df["RRT_Score_error"].min()
max_val = df["RRT_Score_error"].max()
bins = np.arange(min_val, max_val + bin_width, bin_width)

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(df["RRT_Score_error"], bins=bins, edgecolor='black')
plt.axvline(percentile, color='red', linestyle='dashed', linewidth=2, label=f'95th Percentile ({percentile:.4f})')
plt.title("Distribution of RRT_Score_error")
plt.xlabel("RRT_Score_error")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("score_data/step_quantile.png", dpi=300)