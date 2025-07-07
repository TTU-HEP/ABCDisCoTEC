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

qcd = "test_data/background_dataset.root"
signal = "test_data/svjl_mMed-3000GeV_mDark-8GeV_rinv-0.5_dataset.root"

branches_to_load = [
    "MET_significance",
    "MET_phi",
    "DeltaEtaJ0J1FatJet",
    "nElectron_miniPFRelIso_all",
    "nMuon_miniPFRelIso_all",
    "RTFatJet",
    "DeltaPhiMinFatJetMET",
    "weights",
    "MET_pt",
]

constraint_observables = ["MET_pt"]

training_variables = [
    "MET_significance",
    "MET_phi",
    "DeltaEtaJ0J1FatJet",
    "nElectron_miniPFRelIso_all",
    "nMuon_miniPFRelIso_all",
    "RTFatJet",
    "DeltaPhiMinFatJetMET",
]

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
qcd_data_train_weights = qcd_data_train["weights"]
qcd_data_test_weights = qcd_data_test["weights"]
qcd_data_val_weights = qcd_data_val["weights"]
signal_data_train_weights = signal_data_train["weights"]
signal_data_test_weights = signal_data_test["weights"]
signal_data_val_weights = signal_data_val["weights"]

# Only keep the training variables
qcd_data_train = qcd_data_train[training_variables]
qcd_data_test = qcd_data_test[training_variables]
qcd_data_val = qcd_data_val[training_variables]
signal_data_train = signal_data_train[training_variables]
signal_data_test = signal_data_test[training_variables]
signal_data_val = signal_data_val[training_variables]

# Normalize the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_train = pd.concat([qcd_data_train, signal_data_train])
columns = data_train.columns
data_train = pd.DataFrame(scaler.fit_transform(data_train), columns=columns)
data_train["label"] = np.concatenate([np.zeros(len(qcd_data_train)), np.ones(len(signal_data_train))])
data_train["weights"] = np.concatenate([qcd_data_train_weights / qcd_data_train_weights.sum(), signal_data_train_weights / signal_data_train_weights.sum()])
data_train_constraint = pd.concat([qcd_data_train_constraint, signal_data_train_constraint])

data_test = pd.concat([qcd_data_test, signal_data_test])
data_test = pd.DataFrame(scaler.transform(data_test), columns=columns)
data_test["label"] = np.concatenate([np.zeros(len(qcd_data_test)), np.ones(len(signal_data_test))])
data_test["weights"] = np.concatenate([qcd_data_test_weights, signal_data_test_weights])
data_test_constraint = pd.concat([qcd_data_test_constraint, signal_data_test_constraint])

data_val = pd.concat([qcd_data_val, signal_data_val])
data_val = pd.DataFrame(scaler.transform(data_val), columns=columns)
data_val["label"] = np.concatenate([np.zeros(len(qcd_data_val)), np.ones(len(signal_data_val))])
data_val["weights"] = np.concatenate([qcd_data_val_weights/qcd_data_val_weights.sum(), signal_data_val_weights/signal_data_val_weights.sum()])
data_val_constraint = pd.concat([qcd_data_val_constraint, signal_data_val_constraint])

# Shuffle the data (not needed if using a sampler)
data_train = data_train.sample(frac=1, random_state=42)
data_train_constraint = data_train_constraint.sample(frac=1, random_state=42)

# Define the model
model = abcdiscotec.make_abcdiscotec_model(
    input_size=len(training_variables),
    architecture=[32, 16, 8, 4],
    use_batchnorm=True,
    flavor="single",
)

# Define the constraints. Here we are enforcing a decorrelation between the DNN output and the MET_pt variable,
# and closure of the ABCD plane defined by the network and MET_pt
constraints = [
    abcdiscotec.DiscoConstraintLambda(
        obs_name_1="dnn_0",
        obs_name_2="MET_pt",
        lambda_weight=0.1,
    ),
    abcdiscotec.ClosureConstraintMDMM(
        obs_name_1="dnn_0",
        obs_name_2="MET_pt",
        n_events_min=10,
        type="max",
        target=0.1,
        symmetrize=True,
        damping=20,
    ),
]

# Define the constraint manager
constraint_manager = abcdiscotec.make_constraint_manager(
    constraints=constraints,
    extra_variables_for_constraints=constraint_observables,
)

# Get the optimizer and MDMM module
optimizer, mdmm_module = abcdiscotec.get_optimizer_and_mdmm_module(
    model=model,
    learning_rate=1e-3,
    constraint_manager=constraint_manager,
)

# Get the DataLoader
training_dataloader = abcdiscotec.get_dataloader(
    dnn_input_data=torch.tensor(data_train[training_variables].values).float(),
    constraint_data=torch.tensor(data_train_constraint.values).float(),
    labels=torch.tensor(data_train["label"].values).float(),
    weights=torch.tensor(data_train["weights"].values).float(),
    batch_size=4196,
    use_sampler=True,
)

validation_dataloader = abcdiscotec.get_dataloader(
    dnn_input_data=torch.tensor(data_val[training_variables].values).float(),
    constraint_data=torch.tensor(data_val_constraint.values).float(),
    labels=torch.tensor(data_val["label"].values).float(),
    weights=torch.tensor(data_val["weights"].values).float(),
    batch_size=4196,
    use_sampler=True,
)

# Train the model
model, history = abcdiscotec.train_model(
    model=model,
    device=torch.device("cpu"),
    training_dataloader=training_dataloader,
    validation_dataloader=validation_dataloader,
    optimizer=optimizer,
    constraint_manager=constraint_manager,
    mdmm_module=mdmm_module,
    n_epochs=10,
    comet_experiment=experiment,
)

# Save the model
abcdiscotec.save_checkpoint(
    "checkpoint",
    model,
    mdmm_module,
    optimizer,
    history=history,
)

# Plot training history
plt.plot(history["bce_0"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("loss.pdf")
plt.close()

for constraint_name in constraint_manager.get_lambda_constraint_names():
    plt.plot(history[constraint_name], label=constraint_name)
plt.xlabel("Epoch")
plt.ylabel("Constraint value")
plt.legend()
plt.savefig("lambda_constraints.pdf")
plt.close()

for constraint_name in constraint_manager.get_mdmm_constraint_names():
    plt.plot(history[constraint_name], label=constraint_name)
plt.xlabel("Epoch")
plt.ylabel("Constraint value")
plt.legend()
plt.savefig("mdmm_constraints.pdf")
plt.close()

# Evaluate the model
scores = abcdiscotec.evaluate_model(
    model=model,
    device=torch.device("cpu"),
    test_data=torch.tensor(data_test[training_variables].values).float(),
)

# Plot the ABCD plane
plt.hist2d(
    scores["dnn_0_out"][(data_test["label"] == 0).values][:,0],
    data_test_constraint["MET_pt"][(data_test["label"] == 0).values],
    bins=100,
    range=[[0, 1], [0, 1200]],
    weights=data_test["weights"][(data_test["label"] == 0).values],
    norm=matplotlib.colors.LogNorm(),
    cmap="viridis",
)
plt.xlabel("DNN 0 output")
plt.ylabel("MET")

plt.savefig("abcd_plane_bkg.pdf")
plt.clf()
plt.close()
plt.hist2d(
    scores["dnn_0_out"][(data_test["label"] == 1).values][:,0],
    data_test_constraint["MET_pt"][(data_test["label"] == 1).values],
    bins=100,
    range=[[0, 1], [0, 1200]],
    weights=data_test["weights"][(data_test["label"] == 1).values],
    norm=matplotlib.colors.LogNorm(),
    cmap="coolwarm",
)
plt.xlabel("DNN 0 output")
plt.ylabel("MET")
plt.savefig("abcd_plane_signal.pdf")