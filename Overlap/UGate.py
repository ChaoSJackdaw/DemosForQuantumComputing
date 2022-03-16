import cirq

class U(cirq.SingleQubitGate):

    def _decompose_(self, qubits):
        return cirq.T(*qubits), cirq.H(*qubits)

    def __str__(self):
        return 'U'