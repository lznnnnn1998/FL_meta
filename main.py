import torch
import random
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor


from base_model import MyNet
from get_dataset import get_fashion_mnist_dataset, partition_data_dirichlet, plot_client_class_distribution
from attack import sign_flip_attack, ALIE_attack, FOE_attack
from defense import coordinateWiseMedianDefense, krum, coordinateWiseTrimmedMeanDefense, NNM_pre_agg



def fed_avg_agg(global_model:MyNet, client_models:list[MyNet], client_num:int, honest_user_list:list):
    last_state_dict = global_model.state_dict()
    for k, v in last_state_dict.items():
        last_state_dict[k] = torch.zeros_like(v)
    for _idx, c_model in enumerate(client_models):
        if _idx not in honest_user_list:
            continue
        client_state_dict = c_model.state_dict()
        for k, v in client_state_dict.items():
            last_state_dict[k] += client_state_dict[k]
    for k, v in last_state_dict.items():
        last_state_dict[k] = v / len(honest_user_list)
    return last_state_dict

def get_gradient(global_model:MyNet, client_models:list[MyNet], client_num:int):
    client_gradients = []
    last_state_dict = global_model.state_dict()
    for i in range(client_num):
        this_state_dict = client_models[i].state_dict()
        this_gradient = dict()
        for k, v in last_state_dict.items():
            this_gradient[k] = this_state_dict[k] - v
        client_gradients.append(this_gradient)
    return client_gradients


def train_one_epoch(train_loader, model, criterion, optimizer, epoch, client_id):
    total_loss = 0
    model.train()
    for i, data in enumerate(train_loader):
        images = data[0]
        label = data[1]
        images = images.to('cuda')
        x_hat = model(images)
        loss = criterion(x_hat, images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss      
    total_loss /= len(train_loader)
    train_psnr = -10 * torch.log10(total_loss)
    print(f"client_id={client_id}, epoch={epoch}, {i}/{len(train_loader)}: psnr = {train_psnr}\n")
    return model

def eval(test_loader, model, criterion):
    total_loss = 0
    model.eval()

    for i, data in enumerate(test_loader):
        images = data[0]
        label = data[1]
        images = images.to('cuda')
        x_hat = model(images)
        loss = criterion(x_hat, images)
        total_loss += loss
    total_loss /= len(test_loader)
    test_psnr = -10 * torch.log10(total_loss)
    return test_psnr
    
if __name__ == "__main__":
    # hyper-param
    iid = False
    client_num = 10
    batch_size = 64
    learning_rate = 1e-4
    weight_decay = 1e-4
    alpha = 0.1 # for non-iid
    lr_drop = 9
    lr_drop_factor = 0.1
    epoch_num = 100
    device = 'cuda'  # cuda for gpu, cpu for cpu
    malicious_user_num = 3
    malicious_user = []
    for i in range(malicious_user_num):
        malicious_user.append(True)
    for i in range(client_num - malicious_user_num):
        malicious_user.append(False)
    # malicious_user = random.choices([True, False], weights=[0.8, 0.2], k=client_num)
    # dataset/dataloader initialization
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_set, test_set = get_fashion_mnist_dataset(transform)
    train_set_len = len(train_set)
    test_set_len = len(test_set)

    train_index = [i for i in range(train_set_len)]
    test_index = [i for i in range(test_set_len)]
    
    client_train_index = []
    client_train_datasets = []
    client_train_dataLoaders = []
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    if iid:
        random.shuffle(train_index)
        _gap = train_set_len//client_num
        for i in range(client_num-1):
            client_train_index.append(train_index[i * _gap: (i+1) * _gap])
        client_train_index.append(train_index[(client_num-1) * _gap:])
        
        for i in range(client_num):    
            _train_dataset = torch.utils.data.Subset(
                train_set,
                client_train_index[i]
            )
            _train_loader = DataLoader(
                _train_dataset,
                batch_size=batch_size,
                shuffle=True,  # 训练集需要打乱
                num_workers=2
            )
            client_train_datasets.append(_train_dataset)
            client_train_dataLoaders.append(_train_loader)
    else:
        client_data_indices_dirichlet = partition_data_dirichlet(train_set, client_num, alpha)

        # plot distribution
        plot_client_class_distribution(client_data_indices_dirichlet, train_set, client_num, 'dirichlet')
        for i in range(client_num):    
            _train_dataset = torch.utils.data.Subset(
                train_set,
                indices=client_data_indices_dirichlet[i]
            )
            _train_loader = DataLoader(
                _train_dataset,
                batch_size=batch_size,
                shuffle=True,  # 训练集需要打乱
                num_workers=2
            )
            client_train_datasets.append(_train_dataset)
            client_train_dataLoaders.append(_train_loader)
    # ------------------initialize dataloader Done------------------


    # ------------------initialize model------------------
    global_model = MyNet()
    global_model = global_model.to(device)
    # Define optimizer and learning scheduler and loss
    client_optimizers = []
    client_lr_schedulers = []
    client_models = []
    for i in range(client_num):
        local_model = MyNet()
        optimizer = torch.optim.Adam([{'params': local_model.parameters(), 'lr': learning_rate},], weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=lr_drop, gamma=lr_drop_factor)
        client_optimizers.append(optimizer)
        client_lr_schedulers.append(lr_scheduler)
        client_models.append(local_model)
    criterion = nn.MSELoss()


    # ------------------start train------------------
    
    for epoch in range(epoch_num):
        for client_id in range(client_num):
            client_models[client_id].load_state_dict(global_model.state_dict())
            client_models[client_id] = client_models[client_id].to(device)
        with ThreadPoolExecutor(max_workers=client_num) as excutor:
            future_id = [excutor.submit(
                train_one_epoch, 
                client_train_dataLoaders[client_id], 
                client_models[client_id],
                criterion,
                client_optimizers[client_id],
                epoch,
                client_id) for client_id in range(client_num)]
            results = [future_id[i].result() for i in range(client_num)]
        for client_id in range(client_num):
            client_models[client_id] = results[client_id]
            client_lr_schedulers[client_id].step()

        # aggregate models
        
        client_gradients = get_gradient(global_model, client_models, client_num)

        # attack
        # client_models = sign_flip_attack(global_model, client_models, malicious_user, client_gradients)
        # client_models = ALIE_attack(global_model, client_models, malicious_user, client_gradients)
        client_models = FOE_attack(global_model, client_models, malicious_user, client_gradients)
        # no defense
        # honest_user_list = [i for i in range(client_num)]
        # last_state_dict = fed_avg_agg(global_model, client_models, client_num, honest_user_list=honest_user_list)


        # pre aggregate
        # client_models = NNM_pre_agg(client_models, malicious_user_num)


        # defense 
        last_state_dict = coordinateWiseMedianDefense(client_models)
        # last_state_dict = coordinateWiseTrimmedMeanDefense(client_models, malicious_user_num)
        # honest_user_list = krum(global_model, client_models, malicious_user_num, client_gradients, krum_param_m=1)
        # last_state_dict = fed_avg_agg(global_model, client_models, client_num, honest_user_list=honest_user_list)

        # update global model
        global_model.load_state_dict(last_state_dict)
        test_psnr = eval(test_loader, global_model, criterion)
        print(f"Epoch = {epoch}, test_psnr = {test_psnr}\n")
        
        # plot 
        test_image = test_loader.dataset[0][0].to(device)
        x_hat = global_model(test_image)
        plt.imshow(x_hat.squeeze(0).detach().cpu().numpy())
        plt.savefig("/home/FL_meta/rec_img.png")
        plt.clf()
        plt.imshow(test_image.squeeze(0).cpu().numpy())
        plt.savefig("/home/FL_meta/ori_img.png")
        plt.clf()