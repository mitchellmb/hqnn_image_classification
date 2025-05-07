import numpy as np
from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import cudaq
from cudaq import spin
from src.config.config import Config


DATA_CONFIG = Config.get('data')
TRAINING_CONFIG = Config.get('training_parameters')


def normalize_rotations(x):
    # Normalizes rotation np.arrays to [-1, 1] range
    max_val = np.max(np.abs(x))
    if max_val != 0:
        x = x / max_val
    return x


class QuantumLayer(nn.Module):
    def __init__(self, qubit_count: int, quantum_layer_args: dict):
        super(QuantumLayer, self).__init__()

        self.hamiltonians = quantum_layer_args.get('hamiltonians', None)
        self.shift = quantum_layer_args.get('shift', 0.1) 
        self.features_per_qubit = quantum_layer_args.get('features_per_qubit', 1) 
        self.quantum_circuit = QuantumFunction(qubit_count, self.hamiltonians, self.features_per_qubit)
        
    def forward(self, input_tensor):
        logits = QuantumFunction.apply(input_tensor, self.quantum_circuit, self.shift)
        return logits
        

class QuantumFunction(Function):
    def __init__(self, qubit_count: int, hamiltonians: list[cudaq.SpinOperator], features_per_qubit: Literal[1, 2]):

        self.qubit_count = qubit_count
        self.hamiltonian = hamiltonians
        self.features_per_qubit = features_per_qubit

        if self.features_per_qubit == 2:
            @cudaq.kernel
            def kernel(ry_angles: np.ndarray, rx_angles: np.ndarray):
                qubits = cudaq.qvector(len(ry_angles))

                h(qubits) # superposition state via Hadamard gate

                # angle encoding of previous linear layer's outputs
                for idx, qubit in enumerate(qubits):
                    ry(ry_angles[idx], qubit)
               
               # entanglement CNOT - rotation - CNOT
                for idx in range(qubit_count - 1):
                    x.ctrl(qubits[idx], qubits[idx+1])
                    rx(rx_angles[idx+1], qubits[idx+1])
                    x.ctrl(qubits[idx], qubits[idx+1])

                # final rotation on qubit not included in prior for loop
                rx(rx_angles[0], qubits[0])

            self.kernel = kernel

        elif self.features_per_qubit == 1:
            @cudaq.kernel
            def kernel(ry_angles: np.ndarray):
                qubits = cudaq.qvector(len(ry_angles))

                # angle encoding of previous linear layer's outputs
                for idx, qubit in enumerate(qubits):
                    ry(ry_angles[idx], qubit)

            self.kernel = kernel

        else:
            raise ValueError("Quantum kernel requires 1 or 2 features per qubit.")


    def run(self, theta_vals: torch.tensor) -> torch.tensor:
        device_input = theta_vals.device
        theta_vals_np = theta_vals.detach().cpu().numpy()

        expectation_values=[]
        for h in self.hamiltonian:

            results=[]
            for i in theta_vals_np:
                ry_angles = np.array(i)[::self.features_per_qubit]
                rx_angles =  np.array(i)[1::self.features_per_qubit]

                # nonlinear encoding, ensuring [1, -1] range for arcsin
                ry_angles = np.arcsin(normalize_rotations(ry_angles))
                rx_angles = np.arcsin(normalize_rotations(rx_angles))

                results.append(cudaq.observe(self.kernel, h, ry_angles, rx_angles))

            expectation_values.append([results[i].expectation() for i in range(len(results))])
        expectation_values = torch.tensor(expectation_values, device=device_input).T

        return expectation_values

    @staticmethod
    def forward(ctx, input_tensor: torch.tensor, quantum_circuit, shift) -> torch.tensor:
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit

        expectation_value = ctx.quantum_circuit.run(input_tensor)

        ctx.save_for_backward(input_tensor, expectation_value)
        return expectation_value
    

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, _ = ctx.saved_tensors
        batch_size, input_dim = input_tensor.shape
        _, num_logits = grad_output.shape

        shift = ctx.shift
        circuit = ctx.quantum_circuit

        # batch process shift+ and shift- tensors to estimate the gradient
        plus_inputs = input_tensor.unsqueeze(1).repeat(1, input_dim, 1)
        minus_inputs = plus_inputs.clone()

        idx = torch.arange(input_dim)
        plus_inputs[:, idx, idx] += shift
        minus_inputs[:, idx, idx] -= shift

        thetas_plus = plus_inputs.reshape(-1, input_dim)
        thetas_minus = minus_inputs.reshape(-1, input_dim)

        exp_plus = circuit.run(thetas_plus)
        exp_minus = circuit.run(thetas_minus)

        # compute gradients with the batched parameter shifts
        grads = (exp_plus - exp_minus) / (2 * shift)
        gradients = grads.view(batch_size, input_dim, num_logits)

        # chain rule contraction with grad_output
        final_grads = torch.einsum('bij,bj->bi', gradients, grad_output)

        return final_grads, None, None
