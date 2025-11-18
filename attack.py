import torch
import numpy as np
from torch.distributions.normal import Normal

from utils import vectorize_weight

def sign_flip_attack(global_model, client_models, malicious_user, client_gradients):
    last_state_dict = global_model.state_dict()
    for client_id, client_model in enumerate(client_models):
        if malicious_user[client_id] == True:
            this_state_dict = client_model.state_dict()
            for k, v in this_state_dict.items():
                this_state_dict[k] = last_state_dict[k] - client_gradients[client_id][k]
            client_model.load_state_dict(this_state_dict)
    return client_models

def ALIE_attack(global_model, client_models, malicious_user, client_gradients):
    N = len(client_models)
    f = np.int32(malicious_user).sum()
    s = torch.tensor(int(N / 2 + 1) - f)
    standard_normal = Normal(loc=0.0, scale=1.0)
    z = standard_normal.icdf((N-s)/N)
    vectorized_params = []
    for i in range(0, len(client_models)):
        local_model_params = client_gradients[i]
        vectorized_weight = vectorize_weight(local_model_params)
        vectorized_params.append(vectorized_weight.unsqueeze(-1))
    
    # concatenate all weights by the last dimension (number of clients)
    honest_user_id = torch.where(torch.tensor(malicious_user, dtype=torch.int32)==0)[0]
    malicious_user_id = torch.where(torch.tensor(malicious_user, dtype=torch.int32)==1)[0]
    vectorized_params = torch.cat(vectorized_params, dim=-1)
    mu = torch.mean(vectorized_params[:, honest_user_id], dim=-1)
    sigma = torch.std(vectorized_params[:, honest_user_id], dim=-1)
    attack_param = mu + z * sigma

    index = 0
    attack_param_dict = global_model.state_dict()
    for k, params in attack_param_dict.items():
        attack_param_ = attack_param[index : index + params.numel()].view(params.size())
        index += params.numel()
        attack_param_dict[k] += attack_param_
    
    for idx in malicious_user_id:
        client_models[idx].load_state_dict(attack_param_dict)
    return client_models

def FOE_attack(global_model, client_models, malicious_user, client_gradients, epsilon=1):
    N = len(client_models)
    f = np.int32(malicious_user).sum()

    vectorized_params = []
    for i in range(0, len(client_models)):
        local_model_params = client_gradients[i]
        vectorized_weight = vectorize_weight(local_model_params)
        vectorized_params.append(vectorized_weight.unsqueeze(-1))
    
    # concatenate all weights by the last dimension (number of clients)
    honest_user_id = torch.where(torch.tensor(malicious_user, dtype=torch.int32)==0)[0]
    malicious_user_id = torch.where(torch.tensor(malicious_user, dtype=torch.int32)==1)[0]
    vectorized_params = torch.cat(vectorized_params, dim=-1)
    mu = torch.mean(vectorized_params[:, honest_user_id], dim=-1)

    attack_param = -mu * epsilon

    index = 0
    attack_param_dict = global_model.state_dict()
    for k, params in attack_param_dict.items():
        attack_param_ = attack_param[index : index + params.numel()].view(params.size())
        index += params.numel()
        attack_param_dict[k] += attack_param_
    
    for idx in malicious_user_id:
        client_models[idx].load_state_dict(attack_param_dict)
    return client_models