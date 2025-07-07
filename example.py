import uproot
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from comet_ml import Experiment

import abcdiscotec

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experiment = Experiment(
    "kpzBaCBjY8CnZbwLdTVTrJVXf",
    project_name="abcdiscotec",
    workspace="rseidita",
)

qcd = "test_data5/background_data.root"
signal = "test_data5/signal_data.root"

branches_to_load = [
    "var0", "var1", "var2", "var3", "var4", "var5", "var6", "var7", "var8", "var9"
]

constraint_observables = ["var0"]

training_variables = branches_to_load[1:]

qcd_file = uproot.open(qcd)
signal_file = uproot.open(signal)

qcd_tree = qcd_file["Events"]
signal_tree = signal_file["Events"]

qcd_data = qcd_tree.arrays(branches_to_load, library="pd")
signal_data = signal_tree.arrays(branches_to_load, library="pd")

# Split the data into training and testing
qcd_data_train = qcd_data.sample(frac=0.8, random_state=42)
qcd_data_test = qcd_data.drop(qcd_data_train.index)
qcd_data_val = qcd_data_train.sample(frac=0.2, random_state=42)
qcd_data_train = qcd_data_train.drop(qcd_data_val.index)

signal_data_train = signal_data.sample(frac=0.8, random_state=42)
signal_data_test = signal_data.drop(signal_data_train.index)
signal_data_val = signal_data_train.sample(frac=0.2, random_state=42)
signal_data_train = signal_data_train.drop(signal_data_val.index)

# Separate the stuff needed for the constraints
qcd_data_train_constraint = qcd_data_train[constraint_observables]
qcd_data_test_constraint = qcd_data_test[constraint_observables]
qcd_data_val_constraint = qcd_data_val[constraint_observables]
signal_data_train_constraint = signal_data_train[constraint_observables]
signal_data_test_constraint = signal_data_test[constraint_observables]
signal_data_val_constraint = signal_data_val[constraint_observables]

# Extract the weights
qcd_data_train_weights = qcd_data_train["var9"]
qcd_data_test_weights = qcd_data_test["var9"]
qcd_data_val_weights = qcd_data_val["var9"]
signal_data_train_weights = signal_data_train["var9"]
signal_data_test_weights = signal_data_test["var9"]
signal_data_val_weights = signal_data_val["var9"]

data_train = pd.concat([qcd_data_train, signal_data_train])
data_train.to_csv("processed_data.csv", index=False) # Save processed data for histogram plots("processed_data.csv", index=False) # Save processed data for histogram plots

# Only keep the training variables
qcd_data_train = qcd_data_train[training_variables]
qcd_data_test = qcd_data_test[training_variables]
qcd_data_val = qcd_data_val[training_variables]
signal_data_train = signal_data_train[training_variables]
signal_data_test = signal_data_test[training_variables]
signal_data_val = signal_data_val[training_variables]

print("Branch names updated to use var0-var9.")
