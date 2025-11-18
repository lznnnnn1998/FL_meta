import torch
import numpy as np

from utils import vectorize_weight

from base_model import MyNet
def krum(global_model, client_models, malicious_user_num, client_gradients, krum_param_m=1):
    N = len(client_models)
    f = malicious_user_num
    vectorized_params = []
    for i in range(0, len(client_models)):
        local_model_params = client_models[i].state_dict()
        vectorized_weight = vectorize_weight(local_model_params)
        vectorized_params.append(vectorized_weight.unsqueeze(-1))
    
    # concatenate all weights by the last dimension (number of clients)
    vectorized_params = torch.cat(vectorized_params, dim=-1)
    dist = vectorized_params.unsqueeze(1).repeat(1, len(client_models), 1) - vectorized_params.unsqueeze(-1).repeat(1, 1, len(client_models))
    dist = torch.sum(dist ** 2, dim=0)

    krum_score = []
    num_neighbors = N - f - 2
    for i in range(N):
        dist_list = []
        for j in range(N):
            if i != j:
                dist_list.append(dist[i][j].item())
        dist_list.sort()
        score = dist_list[0 : num_neighbors]
        krum_score.append(sum(score))
    score_index = torch.argsort(torch.Tensor(krum_score)).tolist()
    benign_user_idx = score_index[0 : krum_param_m] # krum_param_m = 1 fro krum, >1 for multi-krum
    return benign_user_idx

def coordinateWiseTrimmedMeanDefense(client_models, malicious_user_num):
    f = malicious_user_num
    vectorized_params = []
    for i in range(0, len(client_models)):
        local_model_params = client_models[i].state_dict()
        vectorized_weight = vectorize_weight(local_model_params)
        vectorized_params.append(vectorized_weight.unsqueeze(-1))
    vectorized_params = torch.cat(vectorized_params, dim=-1)
    norm_list = torch.norm(vectorized_params, p=2, dim=0)
    indices = torch.argsort(norm_list)
    assert 2 * f < len(client_models)
    remained_indices = indices[f:-f]
    vec_median_params = torch.mean(vectorized_params[:, remained_indices], dim=-1)
    index = 0
    averaged_params = client_models[0].state_dict()
    for k, params in averaged_params.items():
        median_params = vec_median_params[index : index + params.numel()].view(params.size())
        index += params.numel()
        averaged_params[k] = median_params
    return averaged_params

def coordinateWiseMedianDefense(client_models):
    vectorized_params = []
    for i in range(0, len(client_models)):
        local_model_params = client_models[i].state_dict()
        vectorized_weight = vectorize_weight(local_model_params)
        vectorized_params.append(vectorized_weight.unsqueeze(-1))
    
    # concatenate all weights by the last dimension (number of clients)
    vectorized_params = torch.cat(vectorized_params, dim=-1)
    vec_median_params = torch.median(vectorized_params, dim=-1).values
    index = 0
    averaged_params = client_models[0].state_dict()
    for k, params in averaged_params.items():
        median_params = vec_median_params[index : index + params.numel()].view(params.size())
        index += params.numel()
        averaged_params[k] = median_params
    return averaged_params

def NNM_pre_agg(client_models, malicious_user_num):
    N = len(client_models)
    f = malicious_user_num
    vectorized_params = []
    for i in range(0, len(client_models)):
        local_model_params = client_models[i].state_dict()
        vectorized_weight = vectorize_weight(local_model_params)
        vectorized_params.append(vectorized_weight.unsqueeze(-1))
    
    # concatenate all weights by the last dimension (number of clients)
    vectorized_params = torch.cat(vectorized_params, dim=-1)
    dist = vectorized_params.unsqueeze(1).repeat(1, len(client_models), 1) - vectorized_params.unsqueeze(-1).repeat(1, 1, len(client_models))
    dist = torch.sum(dist ** 2, dim=0)
    new_client_models = [MyNet() for _ in range(N)]
    for model in new_client_models:
        model.to(device=vectorized_weight.device)
    for i in range(N):
        indices = torch.argsort(dist[i])[:-f]
        empty_state_dict = client_models[0].state_dict()
        for k,v in empty_state_dict.items():
            empty_state_dict[k] = torch.zeros_like(v)
        
        for k in empty_state_dict.keys():
            for idx in indices:
                empty_state_dict[k] += client_models[idx].state_dict()[k]
            empty_state_dict[k] /= len(indices)
        new_client_models[i].load_state_dict(empty_state_dict)

    return new_client_models