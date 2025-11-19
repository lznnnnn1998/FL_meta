from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

def partition_data_dirichlet(dataset:datasets.FashionMNIST, num_clients, alpha):
    """
    Creates a non-IID partition of a dataset using a Dirichlet distribution.
    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
        alpha: Concentration parameter for Dirichlet distribution (smaller = more non-IID)
    Returns:
        client_indices: dict where key=client_id, value=indices of dataset for that client.
    """
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))
    num_samples = len(targets)
    
    # 1. Get indices for each class
    class_indices = [np.where(targets == i)[0] for i in range(num_classes)]
    
    # 2. For each class, split the indices among clients based on Dirichlet distribution
    client_indices = {i: [] for i in range(num_clients)}
    proportions = np.random.dirichlet(np.repeat(alpha, num_clients), size=num_classes)
    
    for c in range(num_classes):
        np.random.shuffle(class_indices[c])
        # Calculate how many samples of this class each client gets
        client_counts = (proportions[c] * len(class_indices[c])).astype(int)
        # Ensure we assign all samples
        client_counts[-1] = len(class_indices[c]) - np.sum(client_counts[:-1])
        
        # Split the class indices and assign them
        splits = np.split(class_indices[c], np.cumsum(client_counts))[:-1]
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())
    
    # Shuffle the indices for each client (optional but good practice)
    for client_id in client_indices:
        np.random.shuffle(client_indices[client_id])
    
    return client_indices

def plot_client_class_distribution(client_data_indices, dataset, num_clients, method_name):
    plt.figure(figsize=(15, 5))
    for client_id in range(num_clients):
        plt.subplot(2, num_clients//2, client_id+1) # Adjust subplot grid as needed
        indices = client_data_indices[client_id]
        targets = np.array(dataset.targets)[indices]
        plt.hist(targets, bins=np.arange(11)-0.5, rwidth=0.8)
        plt.title(f'Client {client_id}')
        plt.ylim(0, 3000) # Set a common y-axis for better comparison
        plt.xticks(range(10))
    plt.suptitle(f'Class Distribution per Client ({method_name})')
    plt.tight_layout()
    plt.show()
    plt.savefig(f'/home/FL_meta/data_distribution_{method_name}.png')

def get_fashion_mnist_dataset(transform):
    train_dataset = datasets.FashionMNIST(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    
    # 下载测试集
    test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    return train_dataset, test_dataset

def get_fashion_mnist_dataloaders(batch_size=64, num_workers=2):
    """
    创建FashionMNIST数据集的训练集和测试集DataLoader
    
    参数:
    batch_size: 每个batch的样本数量
    num_workers: 数据加载的线程数
    """
    # 定义数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将PIL图像或numpy数组转换为Tensor
    ])
    
    # 下载训练集
    train_dataset = datasets.FashionMNIST(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    
    # 下载测试集
    test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练集需要打乱
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试集不需要打乱
        num_workers=num_workers
    )
    
    return train_loader, test_loader