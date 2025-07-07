import torch
from typing import Union

import mdmm
from tqdm import tqdm
from .models import SimpleNet, ABCDiscoNet
from .constraint_manager import ConstraintManager, LambdaConstraint, MdmmConstraint
from .dataset import ABCDDataSet
from ._utilities import plot_abcd_plane
import matplotlib.pyplot as plt

def make_abcdiscotec_model(
        input_size:int,
        architecture:Union[list[int],tuple[int]],
        use_batchnorm:bool=True,
        flavor:str="double",
        )->torch.nn.Module:
    """
    Make a simple neural network model for the ABCD method.

    Args:
        input_size (int): The size of the input.
        architecture (Union[list[int],tuple[int]]): The architecture of the neural network.
        use_batchnorm (bool): Whether to use batch normalization.
        flavor (str): The flavor of the model, either "single" or "double".

    Returns:
        torch.nn.Module: The neural network model.
    """

    assert flavor in ["single", "double"], "Flavor must be either 'single' or 'double'."

    networks = []

    networks.append(SimpleNet(input_size, architecture, use_batchnorm))
    
    if flavor == "double":
        networks.append(SimpleNet(input_size, architecture, use_batchnorm))

    return ABCDiscoNet(networks)


def save_checkpoint(
        path:str,
        model:torch.nn.Module,
        mdmm_module:mdmm.MDMM,
        optimizer:torch.optim.Optimizer,
        history:dict,
        lr_scheduler:torch.optim.lr_scheduler.LRScheduler=None,
        ):
    """
    Save a model checkpoint. The checkpoint includes the model, the MDMM module,
    the optimizer, the learning rate scheduler, and the training history.

    Args:
        model (torch.nn.Module): The model to save.
        mdmm_module (mdmm.MDMM): The MDMM module.
        optimizer (torch.optim.Optimizer): The optimizer.
        lr_scheduler (torch.optim.lr_scheduler.LRScheduler): The learning rate scheduler.
        history (dict): The training history.
        path (str): The path to save the model to.
    """

    chechpoint = {
        "model":model.state_dict(),
        "mdmm_module":mdmm_module.state_dict(),
        "optimizer":optimizer.state_dict(),
        "history":history,
        "lr_scheduler":lr_scheduler.state_dict() if lr_scheduler else None,
    }
    
    torch.save(chechpoint, path+".pt")


def load_checkpoint(
        path:str,
        model:torch.nn.Module,
        mdmm_module:mdmm.MDMM,
        optimizer:torch.optim.Optimizer,
        lr_scheduler:torch.optim.lr_scheduler.LRScheduler=None,
        device:torch.device=torch.device("cpu"),
        )->tuple[torch.nn.Module,mdmm.MDMM,torch.optim.Optimizer,dict,torch.optim.lr_scheduler.LRScheduler]:
    """
    Load a model checkpoint. The checkpoint includes the model, the MDMM module,
    the optimizer, the learning rate scheduler, and the training history.

    Args:
        path (str): The path to the model checkpoint.
        model (torch.nn.Module): The model to load the checkpoint into.
        mdmm_module (mdmm.MDMM): The MDMM module to load the checkpoint into.
        optimizer (torch.optim.Optimizer): The optimizer to load the checkpoint into.
        lr_scheduler (torch.optim.lr_scheduler.LRScheduler): The learning rate scheduler to load the checkpoint into.
        device (torch.device): The device to load the model to.

    Returns:
        tuple[torch.nn.Module,mdmm.MDMM,torch.optim.Optimizer,dict,torch.optim.lr_scheduler.LRScheduler]: 
        The model, MDMM module, optimizer, history, and learning rate scheduler
    """

    checkpoint = torch.load(path+".pt", weights_only=True, map_location=device)
    model.load_state_dict(checkpoint["model"])
    mdmm_module.load_state_dict(checkpoint["mdmm_module"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if lr_scheduler: lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    return model, mdmm_module, optimizer, checkpoint["history"], lr_scheduler


def get_dataloader(
        dnn_input_data:torch.tensor,
        constraint_data:torch.tensor,
        labels:torch.tensor,
        weights:torch.tensor,
        batch_size:int,
        use_sampler:bool=True,
        )->torch.utils.data.DataLoader:
    """
    Get a DataLoader for the ABCD dataset.

    Args:
        data (torch.tensor): The data.
        constraint_obs (torch.tensor): The constraint observations.
        labels (torch.tensor): The labels.
        weights (torch.tensor): The weights.
        batch_size (int): The batch size.
        use_sampler (bool): Whether to use a sampler.

    Returns:
        torch.utils.data.DataLoader: The DataLoader.
    """
    dataset = ABCDDataSet(dnn_input_data, constraint_data, labels, weights)
    if use_sampler:
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def make_constraint_manager(
        constraints:list[Union[LambdaConstraint,MdmmConstraint]],
        extra_variables_for_constraints:list[str],
        )->ConstraintManager:
    """
    Make a constraint manager.

    Args:
        constraints (list[Union[LambdaConstraint,MdmmConstraint]]): The constraints.

    Returns:
        ConstraintManager: The constraint manager.
    """
    constraint_manager = ConstraintManager(extra_variables_for_constraints)
    for constraint in constraints:
        constraint_manager.add_constraint(constraint)

    return constraint_manager


def get_optimizer_and_mdmm_module(
        model:torch.nn.Module,
        learning_rate:float,
        constraint_manager:ConstraintManager,
        )->tuple:
    """
    Get the optimizer and MDMM module.

    Args:
        model (torch.nn.Module): The model.
        learning_rate (float): The learning rate.
        constraint_manager (ConstraintManager): The constraint manager.

    Returns:
        tuple: The optimizer and MDMM module.
    """
    optimizer, mdmm_module = constraint_manager.make_mdmm_optimizer_and_module(model, learning_rate)

    return optimizer, mdmm_module


def train_model(
        model:ABCDiscoNet,
        device:torch.device,
        training_dataloader:torch.utils.data.DataLoader,
        validation_dataloader:torch.utils.data.DataLoader,
        optimizer:torch.optim.Optimizer,
        constraint_manager:ConstraintManager,
        mdmm_module:mdmm.MDMMReturn,
        n_epochs:int,
        comet_experiment=None,
        history:dict=None,
        )->tuple[torch.nn.Module,dict]:
    """
    Train the model.

    Args:
        model (ABCDiscoNet): The model to train.
        training_dataloader (torch.utils.data.DataLoader): The DataLoader for the training data.
        validation_dataloader (torch.utils.data.DataLoader): The DataLoader for the validation data.
        optimizer (torch.optim.Optimizer): The optimizer.
        constraint_manager (ConstraintManager): The constraint manager.
        mdmm_module (torch.nn.Module): The MDMM module.
        n_epochs (int): The number of epochs.
        comet_experiment (comet_ml.Experiment): The Comet experiment.
        history (dict): The training history (in case of resumed training).

    Returns:
        tuple[torch.nn.Module,dict]: The trained model and the training history.
    """

    # Initialize history
    if history is None:
        history = {}
        for i in range(len(model)): history[f"bce_{i}"] = []
        for i in range(len(model)): history[f"val_bce{i}"] = []
        for constraint_name in constraint_manager.get_lambda_constraint_names():
            history[constraint_name] = []
        for constraint_name in constraint_manager.get_mdmm_constraint_names():
            history[constraint_name] = []

    # If history is loaded from a checkpoint, check if all constraints are present and that
    # the history is consistent with the model
    else:
        for i in range(len(model)):
            if f"bce{i}" not in history:
                raise ValueError(f"History is missing bce_{i}, please check that the correct checkpoint is being loaded.")
            if f"val_bce{i}" not in history:
                raise ValueError(f"History is missing val_bce{i}, please check that the correct checkpoint is being loaded.")
        for constraint_name in constraint_manager.get_lambda_constraint_names():
            if constraint_name not in history:
                raise ValueError(f"Constraint {constraint_name} is missing from the history, please check " +
                                 "that the correct checkpoint is being loaded.")
        for constraint_name in constraint_manager.get_mdmm_constraint_names():
            if constraint_name not in history:
                raise ValueError(f"Constraint {constraint_name} is missing from the history, please check " +
                                 "that the correct checkpoint is being loaded.")

    # Training loop
    model.to(device)
    for epoch in range(n_epochs):
        model.train()
        n_steps = len(training_dataloader)
        for k in history.keys(): history[k].append(0.)
        for data, constraint_obs, labels, weights in tqdm(
            training_dataloader,
            desc=f"Epoch {epoch+1}/{n_epochs}",
            ):

            # Move data to device
            data = data.to(device)
            constraint_obs = constraint_obs.to(device)
            labels = labels.to(device)
            weights = weights.to(device)

            optimizer.zero_grad()
            dnn_outputs = model(data)

            bce = [torch.nn.BCELoss()(dnn_out[:,0], labels.float()) for dnn_out in dnn_outputs]
            
            loss = sum(bce)

            for i in range(len(model)): history[f"bce_{i}"][-1] += bce[i].detach().cpu().item()

            # Add lambda constraints
            for constraint, constraint_name in zip(
                constraint_manager.get_lambda_constraints(),
                constraint_manager.get_lambda_constraint_names()
                ):
                const = constraint(
                    dnn_outputs,
                    constraint_obs,
                    constraint_manager.constraint_obs,
                    weights,
                    labels,
                    )
                loss += const
                history[constraint_name][-1] += const.detach().cpu().item()

            # Add MDMM constraints
            mdmm_return = mdmm_module(
                loss,
                [(dnn_outputs, constraint_obs, constraint_manager.constraint_obs, weights, labels)
                 for _ in constraint_manager.get_mdmm_constraints()],
            )

            mdmm_return.value.backward()
            optimizer.step()

            for constraint_name, value in zip(
                constraint_manager.get_mdmm_constraint_names(),
                mdmm_return.fn_values
                ):
                history[constraint_name][-1] += value.detach().cpu().item()

        for k in history.keys():
            history[k][-1] /= n_steps
            if comet_experiment: comet_experiment.log_metric(k, history[k][-1], step=epoch)

        # Validation loop
        model.eval()
        bce_val = [0. for _ in range(len(model))]
        with torch.no_grad():
            for data, constraint_obs, labels, weights in tqdm(validation_dataloader, desc="Validating"):
                data = data.to(device)
                constraint_obs = constraint_obs.to(device)
                labels = labels.to(device)
                weights = weights.to(device)

                dnn_outputs_val = model(data)

                for i, dnn_out in enumerate(dnn_outputs_val):
                    bce_val[i] += torch.nn.BCELoss()(dnn_out[:,0], labels.float()).detach().cpu().item()

            for i in range(len(model)):
                bce = bce_val[i] / len(validation_dataloader)
                history[f"val_bce{i}"][-1] = bce

                if comet_experiment:
                    comet_experiment.log_metric(f"val_bce{i}", bce, step=epoch)

    return model, history


def evaluate_model(
        model:torch.nn.Module,
        test_data:torch.tensor,
        device:torch.device,
        )->dict:
    """
    Evaluate the model.

    Args:
        model (torch.nn.Module): The model.
        test_data (torch.tensor): The test data.

    Returns:
        dict: The evaluation results.
    """
    model.to(device)
    model.eval()

    test_data = test_data.to(device)

    with torch.no_grad():
        dnn_outputs = model(test_data)
    
    return {f"dnn_{i}_out":dnn_outputs[i] for i in range(len(model))}