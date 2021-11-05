import cirq
import numpy as np

# 制备2量子比特的|b>初态
a = cirq.NamedQubit("0")
b = cirq.NamedQubit("1")
circuit = cirq.Circuit()


# 二叉树形式制备对应量子态
def pre_circuit():
    circuit.append(cirq.ry(np.pi * 0.5).on(a))
    circuit.append(cirq.ry(np.pi * 0.8).on(b).controlled_by(a))
    circuit.append(cirq.X(a))
    circuit.append(cirq.ry(np.pi * 0.6).on(b).controlled_by(a))
    circuit.append(cirq.X(a))


pre_circuit()
result = cirq.Simulator().simulate(circuit)
print(circuit)
print(result.dirac_notation())

c = cirq.NamedQubit("0")
d = cirq.NamedQubit("1")
e = cirq.NamedQubit("2")
circuit2 = cirq.Circuit()


# 使用辅助量子比特制备|b>
def a_pre_circuit():
    circuit2.append(cirq.H(d))
    circuit2.append(cirq.H(e))
    circuit2.append(cirq.ry(2 * np.arcsin(0.42)).on(c).controlled_by(d, e))
    circuit2.append(cirq.X(e))
    circuit2.append(cirq.ry(2 * np.arcsin(0.57)).on(c).controlled_by(d, e))
    circuit2.append(cirq.X(d))
    circuit2.append(cirq.X(e))
    circuit2.append(cirq.ry(2 * np.arcsin(0.22)).on(c).controlled_by(d, e))
    circuit2.append(cirq.X(e))
    circuit2.append(cirq.ry(2 * np.arcsin(0.67)).on(c).controlled_by(d, e))


a_pre_circuit()
result = cirq.Simulator().simulate(circuit2)
print(circuit2)
print(result.dirac_notation())

