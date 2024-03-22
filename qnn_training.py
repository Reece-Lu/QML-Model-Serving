import torch
import torch.optim as optim
from torch.nn import Module, Linear, CrossEntropyLoss, Conv2d, Dropout2d, MaxPool2d, Flatten
from torch.nn.functional import relu
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
import numpy as np


class HNet(Module):
    def __init__(self, qnn):
        super(HNet, self).__init__()
        self.conv1 = Conv2d(3, 6, kernel_size=5)
        self.conv2 = Conv2d(6, 16, kernel_size=5)
        self.dropout = Dropout2d()
        self.fc1 = Linear(16 * 5 * 5, 120)
        self.fc2 = Linear(120, 2)  # 2-dimensional input to QNN
        self.qnn = TorchConnector(
            qnn)  # Apply torch connector, weights chosen uniformly at random from interval [-1,1].
        self.fc3 = Linear(1, 1)  # 1-dimensional output from QNN

    def forward(self, x):
        x = relu(self.conv1(x))
        x = MaxPool2d(2)(x)
        x = relu(self.conv2(x))
        x = MaxPool2d(2)(x)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = relu(self.fc1(x))
        x = self.fc2(x)
        x = self.qnn(x)  # apply QNN
        x = self.fc3(x)
        return torch.cat((x, 1 - x), -1)


def create_qnn():
    feature_map = ZZFeatureMap(2)
    ansatz = RealAmplitudes(2, reps=1)
    qc = QuantumCircuit(2)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    qnn = EstimatorQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True
    )
    return qnn


def get_data_loaders(batch_size, n_samples):
    # Use pre-defined torchvision function to load CIFAR10 train data
    X_train = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])
    )

    X_train.targets = torch.Tensor(X_train.targets)

    # Filter out labels, leaving only labels 0 and 1
    idx = np.append(
        np.where(X_train.targets == 0)[0][:n_samples], np.where(X_train.targets == 1)[0][:n_samples]
    )

    X_train.data = X_train.data[idx]
    X_train.targets = X_train.targets[idx]

    # Define torch dataloader with filtered data
    train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)

    # Same process for the test data
    X_test = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])
    )

    X_test.targets = torch.Tensor(X_test.targets)
    idx = np.append(
        np.where(X_test.targets == 0)[0][:n_samples], np.where(X_test.targets == 1)[0][:n_samples]
    )

    X_test.data = X_test.data[idx]
    X_test.targets = X_test.targets[idx]

    test_loader = DataLoader(X_test, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def train_model(model, train_loader, epochs):
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    loss_func = CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, target.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")


def main():
    batch_size = 1
    n_samples = 300  # 我们将只关注前100个样本
    epochs = 15 # 运行20个训练周期
    train_loader, test_loader = get_data_loaders(batch_size, n_samples)

    # 创建量子神经网络
    qnn = create_qnn()

    # 初始化模型
    model = HNet(qnn)

    # 训练模型
    train_model(model, train_loader, epochs)

    # 测试模型性能
    correct = 0
    total_loss = []
    loss_func = CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            output = model(data)
            if len(output.shape) == 1:
                output = output.reshape(1, *output.shape)

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            loss = loss_func(output, target.long())
            total_loss.append(loss.item())

    print(
        "Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%".format(
            sum(total_loss) / len(total_loss), 100. * correct / len(test_loader.dataset)
        )
    )

    # 保存训练好的模型
    torch.save(model.state_dict(), "model_hybrid.pt")


if __name__ == '__main__':
    main()

