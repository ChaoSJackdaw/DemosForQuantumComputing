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
args = []
for i in range(3):
    args.append(random.random() * np.pi - np.pi / 2)

label: int


class QuantumCircuit:
    """
    This class provides a simple interface for interaction
    with the quantum circuit
    """
    def __init__(self, n_qubits, backend, shots):
        # --- Circuit definition ---
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = [i for i in range(n_qubits)]
        self.theta0 = qiskit.circuit.Parameter('theta0')
        self.theta1 = qiskit.circuit.Parameter('theta1')
        self.theta2 = qiskit.circuit.Parameter('theta2')
        self.circuit.h(all_qubits)
        self.circuit.barrier()
        self.circuit.ry(self.theta0, all_qubits[0])
        self.circuit.ry(self.theta1, all_qubits[1])
        self.circuit.ry(self.theta2, all_qubits[2])
        # self.circuit.cz(all_qubits[0], all_qubits[1])
        # self.circuit.cz(all_qubits[1], all_qubits[2])
        # self.circuit.ry(self.theta0, all_qubits[0])
        # self.circuit.ry(self.theta1, all_qubits[1])
        # self.circuit.ry(self.theta2, all_qubits[2])
        # self.circuit.cz(all_qubits[0], all_qubits[1])
        # self.circuit.cz(all_qubits[1], all_qubits[2])
        self.circuit.save_statevector()
        # ---------------------------
        self.backend = backend
        self.shots = shots

    def run(self, thetas):
        t_qc = transpile(self.circuit, self.backend)
        binds_list = [{self.theta0: thetas[0], self.theta1: thetas[1], self.theta2: thetas[2]}]
        qobj = assemble(t_qc, shots=self.shots, parameter_binds=binds_list)
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        expectation = [result['001'], result['010'], result['100']]
        expectation = expectation / sum(expectation)
        return np.array([expectation])


class HybridFunction(Function):

    @staticmethod
    def forward(ctx, input, quantum_circuit, shift):
        """ Forward pass computation """
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit
        expectation_z = ctx.quantum_circuit.run(input[0].tolist())
        result = torch.as_tensor([expectation_z])
        ctx.save_for_backward(input, result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        """ Backward pass computation """
        input, expectation_z = ctx.saved_tensors
        input_list = np.array(input.tolist())
        # threshold = 0.002
        # pre_result = pre_out[0][label].item()
        # res = 1.0
        # while res >= threshold:
        #     for i in range(len(args)):
        #         args[i] = args[i] + ctx.shift
        #         expectation_right = ctx.quantum_circuit.run(input_list[0])
        #         args[i] = args[i] - 2 * ctx.shift
        #         expectation_left = ctx.quantum_circuit.run(input_list[0])
        #         args[i] += ctx.shift + 0.001 * (expectation_right[0][label] - expectation_left[0][label])
        #         args[i] %= 2 * np.pi
        #
        #     current_res = ctx.quantum_circuit.run(input_list[0])
        #     current_res = current_res[0][label]
        #     res = current_res - pre_result
        #     pre_result = current_res
        gradients = []
        for i in range(3):
            """ parameter shift rule """
            theta_right = [input_list[0][0], input_list[0][1], input_list[0][2]]
            theta_left = [input_list[0][0], input_list[0][1], input_list[0][2]]
            theta_right[i] += ctx.shift
            theta_left[i] -= ctx.shift
            expectation_right = ctx.quantum_circuit.run(theta_right)
            expectation_left = ctx.quantum_circuit.run(theta_left)
            gradient = expectation_right[0][label] - expectation_left[0][label]
            gradients.append(gradient)

        # gradients = np.array([gradients]).T
        gradients = np.array([item.detach().numpy() for item in expectation_z])
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
        self.quantum_circuit = QuantumCircuit(3, backend, shots)
        self.shift = shift

    def forward(self, input):
        return HybridFunction.apply(input, self.quantum_circuit, self.shift)


# 训练集
n_samples = 100
X_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

idx = np.append(np.where(X_train.targets == 0)[0][:n_samples],
                np.append(np.where(X_train.targets == 1)[0][:n_samples],
                          np.where(X_train.targets == 2)[0][:n_samples])
                )

X_train.data = X_train.data[idx]
X_train.targets = X_train.targets[idx]

# 测试集
n_samples = 50
X_test = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
idx = np.append(np.where(X_train.targets == 0)[0][:n_samples],
                np.append(np.where(X_train.targets == 1)[0][:n_samples],
                          np.where(X_train.targets == 2)[0][:n_samples]
                          )
                )

X_test.data = X_train.data[idx]
X_test.targets = X_train.targets[idx]

train_loader = torch.utils.data.DataLoader(X_train, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(X_test, batch_size=1, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 二维卷积层，输入通道为1，输出通道为6，卷积核大小为5
        # 28*28 /right 24*24
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
        self.fc2 = nn.Linear(64, 3)
        """
            Hybird(self, backend, shots, shift)
            backend -> qiskit.Aer.get_backend('aer_simulator')
            shots -> 100
            shift -> pi/2
        """
        self.hybrid = Hybrid(qiskit.Aer.get_backend('aer_simulator'), 100, np.pi / 1000)

    # CNN+QNN
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        # 将二维数据变为一维
        x = x.view(1, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.hybrid(x)
        return x[0]


model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.004)
loss_func = nn.NLLLoss()

epoch = 0

x_data = []
y_data = [0]
model.train()
total_loss = []

while epoch <= 20:
    epoch += 1
    x_data.append(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        label = target.item()
        loss.backward()
        optimizer.step()
        output = model(data)
        total_loss.append(loss.item())

    y_data.append(sum(total_loss) / len(total_loss))
    print('Training [{:.0f}]\tValue of Loss Function: {:.4f}'.format(epoch, y_data[-1]))


plt.plot(x_data, y_data[1:], "k")
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
