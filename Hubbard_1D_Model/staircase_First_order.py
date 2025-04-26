import pennylane as qml
from pennylane import numpy as np
import time

def load_pauli_terms(filename):
    terms = []
    with open(filename, 'r') as f:
        for line in f:
            coeff_str, pauli_str = line.strip().split(',')
            coeff = complex(coeff_str.strip().strip('()'))
            pauli_str = pauli_str.strip()
            assert len(pauli_str) == 11

            terms.append((coeff, pauli_str))

    return terms

filename = 'Hubbard_1D_Model/First_order.txt'
terms = load_pauli_terms(filename)

n_qubits = 11
print(f'N qubits: {n_qubits}')
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def apply_trotter_step(dt):
    for coeff, pauli_str in terms:
        I_idx = []
        for i, p in enumerate(pauli_str):
            if p == 'I':
                I_idx.append(i)
            elif p == 'X':
                qml.Hadamard(wires=i)
            elif p == 'Y':
                qml.RZ(-np.pi / 2, wires=i)
                qml.Hadamard(wires=i)
            elif p == 'Z':
                continue
            else:
                raise ValueError(f"Invalid Pauli character: {p}")
    
        effi_list =  [x for x in range(11) if x not in I_idx]
        print(f"Processing term: {pauli_str}, Active qubits: {effi_list}")

        if len(effi_list) > 1:
            for i in range(len(effi_list)-1):
                qml.CNOT(wires=[effi_list[i], effi_list[i + 1]])

        if len(effi_list) >= 1:
            qml.RZ(2 * coeff * dt, wires=effi_list[-1])

        if len(effi_list) > 1:
            for i in reversed(range(len(effi_list)-1)):
                qml.CNOT(wires=[effi_list[i], effi_list[i + 1]])
        
        for i in effi_list:
            if pauli_str[i] == 'X':
                qml.Hadamard(wires=i)
            elif pauli_str[i] == 'Y':
                qml.Hadamard(wires=i)
                qml.RZ(np.pi / 2, wires=i)
            elif pauli_str[i] == 'Z':
                continue
    return qml.state()
    

def time_evolution(t, num_trotter_steps):
    dt = t / num_trotter_steps
    for step in range(num_trotter_steps):
        print(f"Applying Trotter step {step+1}/{num_trotter_steps}")
        apply_trotter_step(dt)
    
    return qml.state()

t = 1
num_trotter_steps = 10

start = time.time()
result= time_evolution(t, num_trotter_steps)
end = time.time()

print("\nFinal state length:", len(result))
print("First 10 amplitudes:", result[:10])
print(f'Time taken: {end - start:.2f} seconds')