import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np


# Define the structure of neural network
class CNet(nn.Module):
    def __init__(self):
        super(CNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 准备数据集
def get_data_loader(batch_size, n_samples):
    # 下载 CIFAR10 数据集
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # 筛选标签为0和1的样本
    train_idx = np.append(np.where(np.array(train_set.targets) == 0)[0][:n_samples],
                          np.where(np.array(train_set.targets) == 1)[0][:n_samples])
    test_idx = np.append(np.where(np.array(test_set.targets) == 0)[0][:n_samples],
                         np.where(np.array(test_set.targets) == 1)[0][:n_samples])

    train_set.data = train_set.data[train_idx]
    train_set.targets = np.array(train_set.targets)[train_idx].tolist()
    test_set.data = test_set.data[test_idx]
    test_set.targets = np.array(test_set.targets)[test_idx].tolist()

    # 创建 DataLoader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


# 训练模型
def train_model(model, train_loader, test_loader, epochs):
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, target.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")

    # 保存模型
    torch.save(model.state_dict(), "model_cnet.pt")


# 主函数
def main():
    batch_size = 1
    n_samples = 100  # We will focus on the first 100 samples.
    # Use fewer iterations in the example to quickly demonstrate, and make sure the network structure is keeping the
    # same as QNN
    epochs = 10

    train_loader, test_loader = get_data_loader(batch_size, n_samples)
    model = CNet()
    train_model(model, train_loader, test_loader, epochs)


if __name__ == "__main__":
    main()

