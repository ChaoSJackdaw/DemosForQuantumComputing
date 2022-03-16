# This is a example to test custom controlled gate
import cirq
import numpy as np


q_c1 = cirq.NamedQubit("c1")
q_c2 = cirq.NamedQubit("c2")
a = cirq.NamedQubit("a")
b = cirq.NamedQubit("b")
circuit = cirq.Circuit()
simulator = cirq.Simulator()


# 定义QPE受控相位门
class U(cirq.SingleQubitGate):
    def __init__(self):
        super(cirq.SingleQubitGate, self)

    def _unitary_(self):
        return np.array([
            [(-1 + 1j) / 2, (1 + 1j) / 2],
            [(1 + 1j) / 2, (-1 + 1j) / 2]
        ])

    def __str__(self):
        return 'U'


class UI(cirq.SingleQubitGate):
    def __init__(self):
        super(cirq.SingleQubitGate, self)

    def _unitary_(self):
        return np.array([
            [(-1 - 1j) / 2, (1 - 1j) / 2],
            [(1 - 1j) / 2, (-1 - 1j) / 2]
        ])

    def __str__(self):
        return 'UI'


class U2(cirq.SingleQubitGate):

    def __init__(self):
        super(cirq.SingleQubitGate, self)

    def _unitary_(self):
        return np.array([[0, -1], [-1, 0]])

    def __str__(self):
        return 'U2'


ui = UI()
u2 = U2()
u = U()
circuit.append(cirq.X(b))


# HHL算法第一部分，QPE
def constructHHL_FirstPart():
    circuit.append(cirq.H(q_c1))
    circuit.append(cirq.H(q_c2))
    circuit.append(u(b).controlled_by(q_c2))
    circuit.append(u2(b).controlled_by(q_c1))
    circuit.append(cirq.SWAP(q_c1, q_c2))
    circuit.append(cirq.H(q_c2))
    circuit.append(cirq.rz(-np.pi / 2).on(q_c1).controlled_by(q_c2))
    circuit.append(cirq.H(q_c1))
    # circuit.append(cirq.SWAP(q_c1, q_c2))


# print-Part1
constructHHL_FirstPart()
print(circuit)
result = simulator.simulate(circuit)
print(result.dirac_notation())


# HHL算法第二部分，受控旋转
def constructHHL_SecondPart():
    # circuit.append(cirq.X(q_c1))
    circuit.append(cirq.ry(np.pi).on(a).controlled_by(q_c1))
    # circuit.append(cirq.X(q_c1))
    # circuit.append(cirq.X(q_c2))
    circuit.append(cirq.ry(np.pi / 3).on(a).controlled_by(q_c2))
    # circuit.append(cirq.X(q_c2))
    circuit.append(cirq.measure(a))


# print-Part2
constructHHL_SecondPart()
print(circuit)
result = simulator.simulate(circuit)
print(result)


# HHL算法第三部分，逆QPE
def constructHHL_ThirdPart():
    # circuit.append(cirq.SWAP(q_c1, q_c2))
    circuit.append(cirq.H(q_c1))
    circuit.append(cirq.rz(np.pi / 2).on(q_c1).controlled_by(q_c2))
    circuit.append(cirq.H(q_c2))
    circuit.append(cirq.SWAP(q_c1, q_c2))
    circuit.append(u2(b).controlled_by(q_c1))
    circuit.append(ui(b).controlled_by(q_c2))
    circuit.append(cirq.H(q_c1))
    circuit.append(cirq.H(q_c2))
    # circuit.append(cirq.measure(b))


constructHHL_ThirdPart()
print(circuit)
result = simulator.simulate(circuit)
print(result)
