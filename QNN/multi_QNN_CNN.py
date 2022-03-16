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
gammas = []
for i in range(9):
    gammas.append(random.random())

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
        self.gamma0 = qiskit.circuit.Parameter('gamma0')
        self.gamma1 = qiskit.circuit.Parameter('gamma1')
        self.gamma2 = qiskit.circuit.Parameter('gamma2')
        self.gamma3 = qiskit.circuit.Parameter('gamma3')
        self.gamma4 = qiskit.circuit.Parameter('gamma4')
        self.gamma5 = qiskit.circuit.Parameter('gamma5')
        self.gamma6 = qiskit.circuit.Parameter('gamma6')
        self.gamma7 = qiskit.circuit.Parameter('gamma7')
        self.gamma8 = qiskit.circuit.Parameter('gamma8')
        self.circuit.h(all_qubits)
        self.circuit.barrier()
        self.circuit.ry(self.theta0, all_qubits[0])
        self.circuit.ry(self.theta1, all_qubits[1])
        self.circuit.ry(self.theta2, all_qubits[2])
        self.circuit.rx(self.gamma0, all_qubits[0])
        self.circuit.rx(self.gamma1, all_qubits[1])
        self.circuit.rx(self.gamma2, all_qubits[2])
        # self.circuit.cx(all_qubits[0], all_qubits[1])
        self.circuit.cx(all_qubits[1], all_qubits[2])
        self.circuit.cx(all_qubits[0], all_qubits[2])
        self.circuit.barrier()
        self.circuit.ry(self.theta0, all_qubits[0])
        self.circuit.ry(self.theta1, all_qubits[1])
        self.circuit.ry(self.theta2, all_qubits[2])
        self.circuit.rx(self.gamma3, all_qubits[0])
        self.circuit.rx(self.gamma4, all_qubits[1])
        self.circuit.rx(self.gamma5, all_qubits[2])
        self.circuit.cx(all_qubits[0], all_qubits[1])
        # self.circuit.cx(all_qubits[1], all_qubits[2])
        self.circuit.cx(all_qubits[0], all_qubits[2])
        self.circuit.barrier()
        self.circuit.rx(self.gamma6, all_qubits[0])
        self.circuit.rx(self.gamma7, all_qubits[1])
        self.circuit.rx(self.gamma8, all_qubits[2])
        self.circuit.save_statevector()
        # ---------------------------
        self.backend = backend
        self.shots = shots

    def run(self, thetas):
        t_qc = transpile(self.circuit, self.backend)
        binds_list = [{self.theta0: thetas[0], self.theta1: thetas[1], self.theta2: thetas[2],
                       self.gamma0: gammas[0], self.gamma1: gammas[1], self.gamma2: gammas[2],
                       self.gamma3: gammas[3], self.gamma4: gammas[4], self.gamma5: gammas[5],
                       self.gamma6: gammas[6], self.gamma7: gammas[7], self.gamma8: gammas[8]
                       }]
        qobj = assemble(t_qc, shots=self.shots, parameter_binds=binds_list)
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        expectation = [result['001'], result['010'], result['100']]
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
        gradients = []
        new_grad = []

        for i in range(9):
            gammas[i] += ctx.shift
            expectation_right = ctx.quantum_circuit.run(input_list[0])
            gammas[i] -= (ctx.shift * 2)
            expectation_left = ctx.quantum_circuit.run(input_list[0])
            gammas[i] += ctx.shift
            gradient = expectation_right[0][label] - expectation_left[0][label]
            new_grad.append(gradient * 0.1)

        for i in range(9):
            gammas[i] = (new_grad[i] * 0.1 + gammas[i]) % (2 * np.pi)

        for i in range(3):
            theta_right = [input_list[0][0], input_list[0][1], input_list[0][2]]
            theta_left = [input_list[0][0], input_list[0][1], input_list[0][2]]
            theta_right[i] += ctx.shift
            theta_left[i] -= ctx.shift
            expectation_right = ctx.quantum_circuit.run(theta_right)
            expectation_left = ctx.quantum_circuit.run(theta_left)
            gradient = expectation_right[0][i] - expectation_left[0][i]
            gradients.append(gradient)

        return torch.tensor([gradients]).float() * grad_output, None, None


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
n_samples = 10
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
        self.hybrid = Hybrid(qiskit.Aer.get_backend('aer_simulator'), 1, np.pi / 2)

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
optimizer = optim.Adam(model.parameters(), lr=0.0005)
loss_func = nn.CrossEntropyLoss()

epoch = 0

x_data = []
y_data = [0]
y_poss = [0]
model.train()
total_loss = []
possibilities = []

while epoch <= 50:
    epoch += 1
    x_data.append(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        label = target.item()
        possbility = output[0]
        possibilities.append(possbility[label] / sum(possbility))
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

    y_data.append(sum(total_loss) / len(total_loss))
    y_poss.append(sum(possibilities) / len(possibilities))
    print('Training [{:.0f}]\tValue of Loss Function: {:.4f}'.format(epoch, y_data[-1]))
    print('Training [{:.0f}]\tValue of Possibility: {:.4f}'.format(epoch, y_poss[-1]))


plt.plot(x_data, y_data[1:], "k")
plt.title('QNN with CNN')
plt.xlabel('Training Iterations')
plt.ylabel('Loss Value')
plt.show()
plt.plot(x_data, y_poss[1:], "k")
plt.title('QNN with CNN')
plt.xlabel('Training Iterations')
plt.ylabel('Possibility of Label')
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
