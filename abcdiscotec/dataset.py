import torch

class ABCDDataSet(torch.utils.data.Dataset):
    '''
    A class to represent the ABCD dataset, including the data, constraint observations, labels, and weights.

    Attributes:
    data (torch.tensor): The data.
    constraint_obs (torch.tensor): The constraint observations.
    labels (torch.tensor): The labels.
    weights (torch.tensor): The weights
    '''
    def __init__(self, data, constraint_obs, labels, weights):
        self.data = data
        self.constraint_obs = constraint_obs
        self.labels = labels
        self.weights = weights

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.constraint_obs[index], self.labels[index], self.weights[index]
    
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