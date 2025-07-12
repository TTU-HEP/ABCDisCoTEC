import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import matplotlib.colors as mcolors
from comet_ml import Experiment
from sklearn.metrics import log_loss, roc_curve, auc
import abcdiscotec

def split_data(df):
    train = df.sample(frac=0.8, random_state=42)
    val = train.sample(frac=0.2, random_state=42)
    train = train.drop(val.index)
    test = df.drop(train.index.union(val.index))
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

class TorchStandardScaler(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, x):
        return (x - self.mean) / self.std

def create_loader(X, constraint_X, y, w, batch_size=4096, shuffle=True):
    dataset = torch.utils.data.TensorDataset(X, constraint_X, torch.tensor(y).float(), torch.tensor(w).float())
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def plot_training(history, key, title, filename):
    plt.figure()
    for k in key if isinstance(key, list) else [key]:
        plt.plot(history[k], label=k)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_signal_background(scores, labels, weights, dnn0="dnn_0_out", dnn1="dnn_1_out", outdir="plots", extraName=""):
    os.makedirs(outdir, exist_ok=True)
    for class_label, title in zip([0, 1], ["Background", "Signal"]):
        fig, ax = plt.subplots(figsize=(6, 6))
        h = ax.hist2d(
            scores[dnn0][labels == class_label][:, 0],
            scores[dnn1][labels == class_label][:, 0],
            bins=100,
            range=[[0, 1], [0, 1]],
            weights=weights[labels == class_label],
            cmap="viridis",
            norm=mcolors.LogNorm() if class_label == 0 else None,
        )
        cb = plt.colorbar(h[3], ax=ax)
        cb.set_label("Number of events", fontsize=12)
        ax.set_xlabel(r"$\mathrm{DNN}_1$", fontsize=14)
        ax.set_ylabel(r"$\mathrm{DNN}_2$", fontsize=14)
        ax.text(0.5, 0.9, title, transform=ax.transAxes, fontsize=14, weight="bold",
                ha="center", bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.5"))
        plt.tight_layout()
        fig.savefig(os.path.join(outdir, f"heatmap_{title.lower()}_{extraName}.pdf"))
        plt.close()

    # Loop over DNN outputs (e.g. dnn0 and dnn1)
    for dnn_idx, dnn_name in enumerate(["dnn_0_out", "dnn_1_out"]):
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Extract scores for background and signal
        bg_mask = labels == 0
        sig_mask = labels == 1
        
        bg_scores = scores[dnn_name][bg_mask][:, 0]
        sig_scores = scores[dnn_name][sig_mask][:, 0]
        
        bg_weights = weights[bg_mask]
        sig_weights = weights[sig_mask]
        
        # Plot histograms
        bins = 100
        ax.hist(bg_scores, bins=bins, range=(0, 1), weights=bg_weights,
                label="Background", histtype='stepfilled', alpha=0.5)
        ax.hist(sig_scores, bins=bins, range=(0, 1), weights=sig_weights,
                label="Signal", histtype='step', linewidth=2)
        
        # Labels and legend
        ax.set_xlabel(f"{dnn_name.upper()} score", fontsize=14)
        ax.set_ylabel("Events (weighted)", fontsize=14)
        ax.legend(fontsize=12)
        ax.set_xlim(0, 1)
        
        plt.tight_layout()
        fig.savefig(os.path.join(outdir, f"{dnn_name}_overlay{extraName}.pdf"))
        plt.close()

    for dnn_idx, dnn_name in enumerate(["dnn_0_out", "dnn_1_out"]):
        # Extract scores and true labels
        dnn_scores = scores[dnn_name][:, 0]
        
        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(labels, dnn_scores, sample_weight=weights)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"{dnn_name.upper()} (AUC = {roc_auc:.3f})", linewidth=2)
        ax.plot([0, 1], [0, 1], "k--", label="Random guess")
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=14)
        ax.set_ylabel("True Positive Rate", fontsize=14)
        ax.set_title("ROC Curve", fontsize=14)
        ax.legend(loc="lower right", fontsize=12)
        
        plt.tight_layout()
        fig.savefig(os.path.join(outdir, f"ROC_{dnn_name}_{extraName}.pdf"))
        plt.close()
        
def train_with_lambdas(disco_lambda, closure_lambda):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment = Experiment(
        api_key="kpzBaCBjY8CnZbwLdTVTrJVXf",
        project_name="abcdiscotec",
        workspace="rseidita",
    )
    experiment.set_name(f"dl={disco_lambda}_cl={closure_lambda}")

    branches = [f"var{i}" for i in range(10)]
    #constraint_obs = ["var0"]
    constraint_obs = []
    training_vars = branches[1:]

    qcd = uproot.open("background_data.root")["Events"].arrays(branches, library="pd")
    signal = uproot.open("signal_data.root")["Events"].arrays(branches, library="pd")
    qcd["weights"] = 1.0
    signal["weights"] = 1.0

    qcd_train, qcd_val, qcd_test = split_data(qcd)
    signal_train, signal_val, signal_test = split_data(signal)

    def prepare_sets(qcd, signal, constraint_obs, training_vars):
        data = pd.concat([qcd[training_vars], signal[training_vars]], ignore_index=True)
        constraints = pd.concat([qcd[constraint_obs], signal[constraint_obs]], ignore_index=True)
        labels = np.concatenate([np.zeros(len(qcd)), np.ones(len(signal))])
        weights = np.concatenate([qcd["weights"], signal["weights"]])
        return data, constraints, labels, weights

    X_train_df, X_train_constraint_df, y_train, w_train = prepare_sets(qcd_train, signal_train, constraint_obs, training_vars)
    X_val_df, X_val_constraint_df, y_val, w_val = prepare_sets(qcd_val, signal_val, constraint_obs, training_vars)
    X_test_df, X_test_constraint_df, y_test, w_test = prepare_sets(qcd_test, signal_test, constraint_obs, training_vars)

    feature_means = X_train_df.mean(axis=0).values
    feature_stds = X_train_df.std(axis=0).values
    scaler = TorchStandardScaler(feature_means, feature_stds)

    X_train = scaler(torch.tensor(X_train_df.values, dtype=torch.float32))
    X_val = scaler(torch.tensor(X_val_df.values, dtype=torch.float32))
    X_test = scaler(torch.tensor(X_test_df.values, dtype=torch.float32))

    X_train_constraint = torch.tensor(X_train_constraint_df.values, dtype=torch.float32)
    X_val_constraint = torch.tensor(X_val_constraint_df.values, dtype=torch.float32)
    X_test_constraint = torch.tensor(X_test_constraint_df.values, dtype=torch.float32)

    model = abcdiscotec.make_abcdiscotec_model(
        input_size=len(training_vars),
        architecture=[100],
        use_batchnorm=True,
        flavor="double",
    )

    constraints = [
        #abcdiscotec.DiscoConstraintLambda("dnn_0", "dnn_1", lambda_weight=disco_lambda),
        #abcdiscotec.ClosureConstraintLambda("dnn_0", "dnn_1", n_events_min=10, lambda_weight=closure_lambda),
        
        abcdiscotec.DiscoConstraintMDMM("dnn_0", "dnn_1", type="max", target=disco_lambda, scale=1.0, damping=1.0),
        abcdiscotec.ClosureConstraintMDMM("dnn_0", "dnn_1", n_events_min=10, type="max", target=closure_lambda, symmetrize=True, scale=1.0, damping=1.0),
        #abcdiscotec.DiscoConstraintMDMM("var0", "dnn_0", type="max", target=0.1, scale=0.1, damping=0),
        #abcdiscotec.DiscoConstraintMDMM("var0", "dnn_1", type="max", target=0.1, scale=0.1, damping=0),
    ]

    constraint_manager = abcdiscotec.make_constraint_manager(
        constraints=constraints,
        extra_variables_for_constraints=constraint_obs,
    )

    optimizer, mdmm_module = abcdiscotec.get_optimizer_and_mdmm_module(
        model=model,
        learning_rate=1e-3,
        constraint_manager=constraint_manager,
    )

    training_loader = create_loader(X_train, X_train_constraint, y_train, w_train)
    validation_loader = create_loader(X_val, X_val_constraint, y_val, w_val, shuffle=False)

    model, history = abcdiscotec.train_model(
        model=model,
        device=device,
        training_dataloader=training_loader,
        validation_dataloader=validation_loader,
        optimizer=optimizer,
        constraint_manager=constraint_manager,
        mdmm_module=mdmm_module,
        n_epochs=100,
        comet_experiment=experiment,
    )

    abcdiscotec.save_checkpoint(f"checkpoint_dl{disco_lambda}_cl{closure_lambda}", model, mdmm_module, optimizer, history=history)
    plot_training(history, ["bce_0", "bce_1", "val_bce0", "val_bce1"], "BCE Loss", f"plots/loss_dl{disco_lambda}_cl{closure_lambda}.pdf")
    plot_training(history, constraint_manager.get_lambda_constraint_names(), "Lambda Constraints", f"plots/lambda_dl{disco_lambda}_cl{closure_lambda}.pdf")
    plot_training(history, constraint_manager.get_mdmm_constraint_names(), "MDMM Constraints", f"plots/mdmm_dl{disco_lambda}_cl{closure_lambda}.pdf")

    val_scores = abcdiscotec.evaluate_model(model, device=device, test_data=X_val)
    p0 = torch.clamp(val_scores["dnn_0_out"], 1e-3, 1 - 1e-3).numpy()
    p1 = torch.clamp(val_scores["dnn_1_out"], 1e-3, 1 - 1e-3).numpy()

    val_bce0 = log_loss(y_val, p0)
    val_bce1 = log_loss(y_val, p1)
    print(f"Manual Validation BCE dnn_0: {val_bce0:.4f}")
    print(f"Manual Validation BCE dnn_1: {val_bce1:.4f}")

    scores = abcdiscotec.evaluate_model(model, device=device, test_data=X_test)
    plot_signal_background(scores, y_test, w_test, extraName=f"dl{disco_lambda}_cl{closure_lambda}")

    return {
        "disco_lambda": disco_lambda,
        "closure_lambda": closure_lambda,
        "final_loss": history["loss"][-1] if "loss" in history else None,
        "accuracy": history["accuracy"][-1] if "accuracy" in history else None
    }

if __name__ == "__main__":
    import itertools
    from multiprocessing import Pool

    disco_lambdas   = [0.01]
    closure_lambdas = [0.01]
    param_grid = list(itertools.product(disco_lambdas, closure_lambdas))

    with Pool(processes=4) as pool:
        results = pool.starmap(train_with_lambdas, param_grid)

    pd.DataFrame(results).to_csv("hyperparam_results_parallel.csv", index=False)
