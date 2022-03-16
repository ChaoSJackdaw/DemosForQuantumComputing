import random
import numpy as np
import matplotlib.pyplot as plt

import qiskit
from qiskit import assemble, transpile

import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# 旋转门初始化参数，backward是静态函数，暂时定义为全局变量
global arg
arg = random.random() * np.pi - np.pi / 2


class QuantumCircuit:
    """
    This class provides a simple interface for interaction
    with the quantum circuit
    """
    def __init__(self, n_qubits, backend, shots):
        # --- Circuit definition ---
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = [i for i in range(n_qubits)]
        self.theta = qiskit.circuit.Parameter('theta')
        self.gamma = qiskit.circuit.Parameter('gamma')
        # self.circuit.h(all_qubits[0])
        self.circuit.barrier()
        self.circuit.h(all_qubits[0])
        self.circuit.ry(self.theta, all_qubits[0])
        self.circuit.rx(self.gamma, all_qubits[0])
        self.circuit.cx(all_qubits[0], all_qubits[1])
        cbit = self.circuit._create_creg(1, "meas")
        self.circuit.add_register(cbit)
        self.circuit.measure(all_qubits[1], cbit)
        # ---------------------------
        self.backend = backend
        self.shots = shots

    def run(self, thetas, gamma):
        t_qc = transpile(self.circuit, self.backend)
        # theta bind to thetas
        qobj = assemble(t_qc, shots=self.shots,
                        parameter_binds=[{self.gamma: gamma, self.theta: thetas[0]}])
        # run quantum circuit
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        counts = np.array(list(result.values()))
        state = np.array(list(result.keys())).astype(float)
        # Compute probabilities for each state
        probabilities = counts / self.shots
        # Get state expectation
        expectation = np.sum(state * probabilities)
        return np.array([expectation])


class HybridFunction(Function):

    @staticmethod
    def forward(ctx, input, quantum_circuit, shift):
        """ Forward pass computation """
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit
        expectation_z = ctx.quantum_circuit.run(input[0].tolist(), arg)
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(input, result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        """ Backward pass computation """
        input, expectation_z = ctx.saved_tensors
        input_list = np.array(input.tolist())
        global arg
        arg_list = np.array([[arg]])
        """ parameter shift rule """
        gamma_right = arg_list + np.ones(input_list.shape) * ctx.shift
        gamma_left = arg_list - np.ones(input_list.shape) * ctx.shift
        theta_right = input_list + np.ones(input_list.shape) * ctx.shift
        theta_left = input_list - np.ones(input_list.shape) * ctx.shift
        gradients = []
        for i in range(len(input_list)):
            # 计算gamma梯度，更新后计算theta
            expectation_gamma_right = ctx.quantum_circuit.run(input_list[i], gamma_left[i][i])
            expectation_gamma_left = ctx.quantum_circuit.run(input_list[i], gamma_right[i][i])
            grad_for_arg = expectation_gamma_right - expectation_gamma_left
            # 更新gamma值
            arg = (arg + grad_for_arg[0] * 0.001) % (2 * np.pi)
            expectation_right = ctx.quantum_circuit.run(theta_right[i], arg_list[i][i])
            expectation_left = ctx.quantum_circuit.run(theta_left[i], arg_list[i][i])
            expectation = expectation_right - expectation_left + grad_for_arg[0]
            # Quantum circuit learning 中有介绍梯度计算
            gradient = torch.tensor([expectation])
            gradients.append(gradient)

        gradients = np.array([gradients]).T
        return torch.tensor([gradients]).float() * grad_output.float(), None, None


# import torch.nn as nn,所有网络模型的基类
class Hybrid(nn.Module):
    """
        Hybrid quantum - classical layer definition
        仿真中使用的参数如下
        backend=qiskit.Aer.get_backend('aer_simulator')
        shots=100
        shift=pi/2
    """
    def __init__(self, backend, shots, shift):
        super(Hybrid, self).__init__()
        self.quantum_circuit = QuantumCircuit(2, backend, shots)
        self.shift = shift

    def forward(self, input):
        return HybridFunction.apply(input, self.quantum_circuit, self.shift)


# Concentrating on the first 100 samples
n_samples = 100

X_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

# 保留MNIST数据集中0和1的数据
idx = np.append(np.where(X_train.targets == 0)[0][:n_samples], np.where(X_train.targets == 1)[0][:n_samples])

# 将0,1数据集覆盖原本的MNIST数据集
X_train.data = X_train.data[idx]
X_train.targets = X_train.targets[idx]

n_samples = 50

X_test = datasets.MNIST(root='./data', train=False, download=True,
                        transform=transforms.Compose([transforms.ToTensor()]))

idx = np.append(np.where(X_test.targets == 0)[0][:n_samples],
                np.where(X_test.targets == 1)[0][:n_samples])

X_test.data = X_test.data[idx]
X_test.targets = X_test.targets[idx]

train_loader = torch.utils.data.DataLoader(X_train, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(X_test, batch_size=1, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 二维卷积层，输入通道为1，输出通道为6，卷积核大小为5
        # self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, 5),
            torch.nn.ReLU()
        )
        # self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 16, 5),
            torch.nn.ReLU()
        )
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)
        """
            Hybird(self, backend, shots, shift)
            backend -> qiskit.Aer.get_backend('aer_simulator')
            shots -> 100
            shift -> pi/2
        """
        self.hybrid = Hybrid(qiskit.Aer.get_backend('aer_simulator'), 100, np.pi / 2)

    # CNN+QNN
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(1, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.hybrid(x)
        return torch.cat((x, 1-x), -1)


model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.005)
loss_func = nn.NLLLoss()

epoch = 0
loss_list2 = [0]
x = []
model.train()
total_loss = []
while epoch <= 50:
    epoch += 1
    x.append(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        # Forward pass
        output = model(data)
        # Calculating loss
        loss = loss_func(output, target)
        # Backward pass
        loss.backward()
        # Optimize the weights
        optimizer.step()
        total_loss.append(loss.item())

    loss_list2.append(sum(total_loss) / len(total_loss))
    print('Training [{:.0f}]\tValue of Loss Function: {:.4f}'.format(epoch, loss_list2[-1]))


plt.plot(x, loss_list2[1:], "k")
plt.title('QNN with CNN')
plt.xlabel('Training Iterations')
plt.ylabel('Loss Value')
plt.show()

model.eval()
with torch.no_grad():
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        output = model(data)

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        loss = loss_func(output, target)
        total_loss.append(loss.item())

    print('Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%'.format(
        sum(total_loss) / len(total_loss),
        correct / len(test_loader) * 100)
    )
