import numpy as np
import pandas as pd
import uproot
import awkward as ak
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

def generate_gaussian_data(n_samples, mean, cov, label):
    """
    Generates multivariate Gaussian-distributed data with specified mean and covariance.
    """
    data = np.random.multivariate_normal(mean, cov, n_samples)
    return {f'var{i}': data[:, i] for i in range(data.shape[1])} | {'label': np.full(n_samples, label)}

# Number of variables and samples
num_vars = 10
num_samples = 100000

# Define means for signal and background
mean_signal = np.linspace(0, 5, num_vars)
mean_background = np.linspace(5, 10, num_vars)

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
cov_signal = create_fully_correlated_cov(num_vars, seed=42)
cov_background = create_fully_correlated_cov(num_vars, seed=24)

signal_data = generate_gaussian_data(num_samples, mean_signal, cov_signal, label=1)
background_data = generate_gaussian_data(num_samples, mean_background, cov_background, label=0)

# Convert to DataFrames
df_signal = pd.DataFrame(signal_data)
df_background = pd.DataFrame(background_data)

# Scale data to [-1, 1] separately
scaler = MinMaxScaler(feature_range=(-1, 1))
features = [col for col in df_signal.columns if col != "label"]
df_signal[features] = scaler.fit_transform(df_signal[features])
df_background[features] = scaler.fit_transform(df_background[features])

# Save to separate ROOT files
with uproot.recreate("signal_data.root") as file_signal:
    file_signal["Events"] = {col: ak.from_numpy(df_signal[col].values) for col in df_signal.columns}

with uproot.recreate("background_data.root") as file_background:
    file_background["Events"] = {col: ak.from_numpy(df_background[col].values) for col in df_background.columns}

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
plt.savefig("correlation_heatmap_signal.png")
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
plt.savefig("correlation_heatmap_background.png")
plt.close()

print("Signal and background data saved to signal_data.root and background_data.root.")
print("Signal correlation heatmap saved as correlation_heatmap_signal.png.")
print("Background correlation heatmap saved as correlation_heatmap_background.png.")
