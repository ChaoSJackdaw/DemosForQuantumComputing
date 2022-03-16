import cirq

qubit = cirq.NamedQubit("|0>")
a = cirq.NamedQubit("a")
b = cirq.NamedQubit("b")
neg_T = cirq.T ** -1
# print(cirq.unitary(neg_T))
# 用于测试初始态的线路
# ops = [cirq.I(qubit), cirq.I(a), cirq.I(b)]
# cir = cirq.Circuit(ops)
simulator = cirq.Simulator()
# result = simulator.simulate(cir)
# print(result)


class U(cirq.SingleQubitGate):

    def _decompose_(self, qubits):
        return cirq.H(*qubits), neg_T(*qubits)

    def __str__(self):
        return 'U'


class NegU(cirq.SingleQubitGate):

    def _decompose_(self, qubits):
        return cirq.T(*qubits), cirq.H(*qubits)

    def __str__(self):
        return 'neg_U'


print("\nABA Circuit:")


def initialcircuit():

    circuit = cirq.Circuit()
    # circuit.append(cirq.X(a))
    circuit.append(cirq.CNOT(a, b))
    circuit.append(cirq.H(qubit))
    circuit.append(cirq.H(a))
    circuit.append(cirq.CNOT(a, b))
    circuit.append(neg_T(b))
    circuit.append(cirq.CNOT(qubit, b))
    circuit.append(cirq.T(b))
    circuit.append(cirq.CNOT(a, b))
    circuit.append(neg_T(b))
    circuit.append(cirq.CNOT(qubit, b))
    circuit.append(cirq.T(a))
    circuit.append(cirq.CNOT(qubit, a))
    circuit.append(cirq.T(qubit))
    circuit.append(neg_T(a))
    circuit.append(cirq.CNOT(qubit, a))
    circuit.append(cirq.H(qubit))
    return circuit


U = U()
NegU = NegU()

def aba():
    circuit = cirq.Circuit()
    circuit.append(U(qubit))
    circuit.append(cirq.CNOT(a, b))
    circuit.append(cirq.CNOT(b, qubit))
    circuit.append(NegU(qubit))
    circuit.append(cirq.CNOT(qubit, a))
    circuit.append(U(qubit))
    circuit.append(cirq.CNOT(b, qubit))
    circuit.append(NegU(qubit))
    return circuit


# Circuit = initialcircuit()
# cirq.measure(qubit, a, b)
Circuit = aba()
result = simulator.simulate(Circuit)
print(Circuit)
print(result.dirac_notation())

