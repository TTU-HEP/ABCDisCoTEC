import numpy as np
import pandas as pd
import uproot
import awkward as ak
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler

def generate_gaussian_data(n_samples, mean, cov, label):
    """
    Generates multivariate Gaussian-distributed data with specified mean and covariance.
    """
    data = np.random.multivariate_normal(mean, cov, n_samples)
    return {f'var{i}': data[:, i] for i in range(data.shape[1])} | {'label': np.full(n_samples, label)}

# Number of variables and samples
num_vars = 10
num_samples = 500000

base_means = np.random.uniform(0, 5, size=num_vars)
offset = np.random.uniform(-2.0, 2.0, size=num_vars)  # adjust spread as needed
mean_background = base_means
mean_signal = base_means + offset

# Create fully correlated covariance matrix with all correlations >= 0.5
def create_fully_correlated_cov(num_vars, base_corr=0.5, random_strength=0.2, seed=None):
    if seed is not None:
        np.random.seed(seed)
    A = np.random.uniform(-random_strength, random_strength, size=(num_vars, num_vars))
    sym_A = (A + A.T) / 2  # Make it symmetric
    np.fill_diagonal(sym_A, 1)  # Variance = 1 on diagonal

    # Mix with a base correlation structure to ensure all correlations are >= base_corr
    cov_matrix = base_corr * np.ones((num_vars, num_vars)) + (1 - base_corr) * sym_A

    # Ensure it's positive definite by adding small multiple of identity matrix
    cov_matrix += 0.1 * np.eye(num_vars)
    return cov_matrix

# Use fully correlated covariance matrices for both signal and background
cov_signal = create_fully_correlated_cov(num_vars, 0.5, 0.2, seed=42)
cov_background = create_fully_correlated_cov(num_vars, 0.5, 0.4, seed=24)

signal_data = generate_gaussian_data(num_samples, mean_signal, cov_signal, label=1)
background_data = generate_gaussian_data(num_samples, mean_background, cov_background, label=0)

# Convert to DataFrames
df_signal = pd.DataFrame(signal_data)
df_background = pd.DataFrame(background_data)

# Scale data to [-1, 1] separately
df_combined = pd.concat([df_signal, df_background], axis=0)
means = df_combined.mean()
stds = df_combined.std()
df_signal = (df_signal - means) / stds
df_background = (df_background - means) / stds

# Add weights
df_signal['weights'] = 1.0
df_background['weights'] = 1.0

# Save to separate ROOT files
# Create output directory for plots
data_dir = "test_data"
os.makedirs(data_dir, exist_ok=True)

with uproot.recreate(f"{data_dir}/signal_data.root") as file_signal:
    file_signal["Events"] = {col: ak.from_numpy(df_signal[col].values) for col in df_signal.columns}

with uproot.recreate(f"{data_dir}/background_data.root") as file_background:
    file_background["Events"] = {col: ak.from_numpy(df_background[col].values) for col in df_background.columns}

# Create output directory for plots
plot_dir = "input_var_plots"
os.makedirs(plot_dir, exist_ok=True)
    
# Plot correlation matrix for signal data
corr_signal = df_signal.drop(columns="label").corr()
sns.set(font_scale=1.1)
plt.figure(figsize=(10, 9))
sns.heatmap(
    corr_signal,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"}
)
plt.title("Correlation Matrix of Signal Input Variables")
plt.tight_layout()
plt.savefig(plot_dir+"/correlation_heatmap_signal.png")
plt.close()

# Plot correlation matrix for background data
corr_background = df_background.drop(columns="label").corr()
plt.figure(figsize=(10, 9))
sns.heatmap(
    corr_background,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"}
)
plt.title("Correlation Matrix of Background Input Variables")
plt.tight_layout()
plt.savefig(plot_dir+"/correlation_heatmap_background.png")
plt.close()

print("Signal and background data saved to signal_data.root and background_data.root.")
print("Signal correlation heatmap saved as correlation_heatmap_signal.png.")
print("Background correlation heatmap saved as correlation_heatmap_background.png.")


# Loop over all shared variables in both dataframes
common_vars = set(df_signal.columns).intersection(df_background.columns)

for var in sorted(common_vars):
    sig_values = df_signal[var].to_numpy()
    bkg_values = df_background[var].to_numpy()

    # Clean data (drop NaN, inf)
    sig_values = sig_values[np.isfinite(sig_values)]
    bkg_values = bkg_values[np.isfinite(bkg_values)]

    # Skip non-numeric or empty
    if sig_values.size == 0 or bkg_values.size == 0:
        continue

    # Plot histogram
    fig, ax = plt.subplots(figsize=(6, 5))
    bins = 50

    ax.hist(bkg_values, bins=bins, density=True, alpha=0.6, label="Background", color="blue", histtype='stepfilled')
    ax.hist(sig_values, bins=bins, density=True, alpha=0.6, label="Signal", color="red", histtype='stepfilled')

    ax.set_title(f"{var} Distribution")
    ax.set_xlabel(var)
    ax.set_ylabel("Normalized Entries")
    ax.legend()
    plt.tight_layout()

    # Save plot
    fig.savefig(os.path.join(plot_dir, f"{var}_hist.png"))
    plt.close()
