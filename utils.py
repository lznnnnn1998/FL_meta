import torch

def vectorize_weight(state_dict:dict[torch.Tensor]):
    weight_list = []
    for k, v in state_dict.items():
        weight_list.append(v.flatten())
    return torch.cat(weight_list)