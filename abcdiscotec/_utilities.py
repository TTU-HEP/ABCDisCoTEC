import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def distance_corr(
        var_1:torch.tensor,
        var_2:torch.tensor,
        normedweight:torch.tensor,
        power=1,
        )->torch.tensor:
    """
    Compute the distance correlation function
    between two variables.

    Args:
        var_1 (torch.tensor): The first variable.
        var_2 (torch.tensor): The second variable.
        normedweight (torch.tensor): The weight matrix.
        power (int): The power of the distance correlation.

    Returns:
        torch.tensor: The distance correlation between the two variables.
    """
    
    # Normalize the weights
    normedweight = normedweight/torch.sum(normedweight)*len(var_1)
    
    xx = var_1.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))
    yy = var_1.repeat(len(var_1),1).view(len(var_1),len(var_1))
    amat = (xx-yy).abs()

    xx = var_2.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))
    yy = var_2.repeat(len(var_2),1).view(len(var_2),len(var_2))
    bmat = (xx-yy).abs()

    amatavg = torch.mean(amat*normedweight,dim=1)
    Amat=amat-amatavg.repeat(len(var_1),1).view(len(var_1),len(var_1))\
        -amatavg.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))\
        +torch.mean(amatavg*normedweight)

    bmatavg = torch.mean(bmat*normedweight,dim=1)
    Bmat=bmat-bmatavg.repeat(len(var_2),1).view(len(var_2),len(var_2))\
        -bmatavg.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))\
        +torch.mean(bmatavg*normedweight)

    ABavg = torch.mean(Amat*Bmat*normedweight,dim=1)
    AAavg = torch.mean(Amat*Amat*normedweight,dim=1)
    BBavg = torch.mean(Bmat*Bmat*normedweight,dim=1)

    if(power==1):
        dCorr=(torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight)))
    elif(power==2):
        dCorr=(torch.mean(ABavg*normedweight))**2/(torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))
    else:
        dCorr=((torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))))**power
    
    return dCorr

def __check_number_of_events(
        var1:torch.tensor,
        var2:torch.tensor,
        rand_cuts:list,
        n_events_min:int,
    )->bool:
    """
    Check if the number of events in each region is above the minimum.

    Args:
        var1 (torch.tensor): The first variable.
        var2 (torch.tensor): The second variable.
        rand_cuts (list): The random cuts.
        n_events_min (int): The minimum number of events in each region.

    Returns:
        bool: True if the number of events in each region is above the minimum, False otherwise.
    """

    cut_var1, cut_var2 = rand_cuts

    NA_diff = torch.sum((torch.sigmoid(100*(var1 - cut_var1))*torch.sigmoid(100*(cut_var2 - var2))))
    NB_diff = torch.sum((torch.sigmoid(100*(var1 - cut_var1))*torch.sigmoid(100*(var2 - cut_var2))))
    NC_diff = torch.sum((torch.sigmoid(100*(cut_var1 - var1))*torch.sigmoid(100*(cut_var2 - var2))))
    ND_diff = torch.sum((torch.sigmoid(100*(cut_var1 - var1))*torch.sigmoid(100*(var2 - cut_var2))))

    check = (
        NA_diff > n_events_min
        and NB_diff > n_events_min
        and NC_diff > n_events_min
        and ND_diff > n_events_min
    )

    return check

def __get_abcd_random_cuts(
        var1:torch.tensor,
        var2:torch.tensor,
        n_events_min:int,
        max_tries:int=1_000_000_000,
        )->list:
    """
    Get random cuts for the closure term in the ABCD plane.

    Args:
        var1 (torch.tensor): The first variable.
        var2 (torch.tensor): The second variable.
        n_events_min (int): The minimum number of events in each region.
        max_tries (int): The maximum number of tries to find a suitable set of cuts.

    Returns:
        list: The random cuts.
    """

    for _ in range(max_tries):
        x_min = torch.quantile(var1, 0.01).item()
        x_max = torch.quantile(var1, 0.99).item()
        rand_cut_x = np.random.uniform(x_min, x_max)
        y_min = torch.quantile(var2, 0.01).item()
        y_max = torch.quantile(var2, 0.99).item()

        rand_cut_y = np.random.uniform(y_min, y_max)
        rand_cuts = [rand_cut_x, rand_cut_y]
        
        check = __check_number_of_events(
            var1,
            var2,
            rand_cuts,
            n_events_min,
        )
        
        if check:
            return rand_cuts
    
    # If we reach this point, we did not find a suitable set of cuts
    raise ValueError(f"Could not find a suitable set of cuts after {max_tries} tries, lower the n_events_min parameter or increase the batch size")

def closure(
        var_1,
        var_2,
        weights,
        labels,
        symmetrize,
        n_events_min=10,
        )->torch.tensor:
    """
    Compute the closure term in the ABCD plane.

    Args:
        var_1 (torch.tensor): The first variable.
        var_2 (torch.tensor): The second variable.
        symmetrize (bool): Whether to symmetrize the closure term.
        weights (torch.tensor): The weights.
        n_events_min (int): The minimum number of events in each region.

    Returns:
        torch.tensor: The closure term.
    """
    
    # Restrict to background events
    var_1 = var_1[labels == 0]
    var_2 = var_2[labels == 0]
    weights = weights[labels == 0]

    cut_var1, cut_var2 = __get_abcd_random_cuts(var_1, var_2, n_events_min)

    # Compute differetiable number of events in each region with sigmoid
    # in order to retain differentiability
    NA_diff = torch.sum((torch.sigmoid(100*(var_1 - cut_var1))*torch.sigmoid(100*(cut_var2 - var_2)))*weights)
    NB_diff = torch.sum((torch.sigmoid(100*(var_1 - cut_var1))*torch.sigmoid(100*(var_2 - cut_var2)))*weights)
    NC_diff = torch.sum((torch.sigmoid(100*(cut_var1 - var_1))*torch.sigmoid(100*(cut_var2 - var_2)))*weights)
    ND_diff = torch.sum((torch.sigmoid(100*(cut_var1 - var_1))*torch.sigmoid(100*(var_2 - cut_var2)))*weights)
    
    # Compute closure_loss
    if symmetrize:
        closure_loss = torch.abs(NA_diff*ND_diff - NB_diff*NC_diff)/(NA_diff*ND_diff + NB_diff*NC_diff)
    else:
        closure_loss = torch.abs(NA_diff*ND_diff - NB_diff*NC_diff)/(NC_diff*NB_diff)
    
    return closure_loss


def get_inputs_for_constraints(
        argument_names:list[str],
        dnn_outputs:tuple[torch.tensor],
        constr_obs:torch.tensor,
        constr_obs_names:list[str],
        weights:torch.tensor,
        labels:torch.tensor,
        ):
    inputs = []
    for obs in argument_names:
        if obs.startswith("dnn_"):
            inputs.append(dnn_outputs[int(obs.split("_")[1])])
        elif obs in constr_obs_names:
            inputs.append(constr_obs[:,constr_obs_names.index(obs)])
        elif obs == "weights":
            inputs.append(weights)
        elif obs == "labels":
            inputs.append(labels)
        else:
            raise ValueError(f"Observable {obs} must be either 'dnn_i', 'weights', 'labels', or one of the constraint observables")

    return inputs


def plot_abcd_plane(
        var_1:torch.tensor,
        var_2:torch.tensor,
        weigths:torch.tensor,
        )->plt.figure:
    """
    Plot the ABCD plane.

    Args:
        var_1 (torch.tensor): The first variable.
        var_2 (torch.tensor): The second variable.

    Returns:
        plt.figure: The figure.
    """
    fig, ax = plt.subplots()
    ax.hist2d(
        var_1.detach().numpy(),
        var_2.detach().numpy(),
        weights=weigths.detach().cpu().numpy(),
        bins=50,
        range=[[0, 1], [0, 1]],
        cmap="viridis",
        norm=matplotlib.colors.LogNorm(),
        )
    ax.set_xlabel("DNN 1 output")
    ax.set_ylabel("DNN 2 output")

    return fig