import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import cudaq
from cudaq import spin
from src.config.config import Config


DATA_CONFIG = Config.get('data')
TRAINING_CONFIG = Config.get('training_parameters')

# device = torch.device('cuda' if torch.cuda.is_available() and TRAINING_CONFIG.get('cuda_device') == 'gpu' else 'cpu')
# cudaq.set_target("qpp-cpu") 


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
    def __init__(self, qubit_count: int, hamiltonians: list[cudaq.SpinOperator], features_per_qubit: int):

        self.qubit_count = qubit_count
        self.hamiltonian = hamiltonians
        self.features_per_qubit = features_per_qubit

        @cudaq.kernel
        def kernel(ry_angles: np.ndarray, rx_angles: np.ndarray):
            qubits = cudaq.qvector(len(ry_angles))

            for idx, qubit in enumerate(qubits):
                ry(ry_angles[idx], qubit)
                ry(rx_angles[idx], qubit)    

            # for i in range(1, qubit_count): # entangle qubits before measuring control-x gate
            #     x.ctrl(qubits[0], qubits[i]) 

        self.kernel = kernel


    def run(self, theta_vals: torch.tensor) -> torch.tensor:
        device_input = theta_vals.device
        theta_vals_np = theta_vals.detach().cpu().numpy()

        expectation_values=[]
        for h in self.hamiltonian:

            results=[]
            for i in theta_vals_np:
                ry_angles = np.array(i)[::self.features_per_qubit]
                rx_angles =  np.array(i)[1::self.features_per_qubit]
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

        device_input = input_tensor.device
        shift = ctx.shift
        circuit = ctx.quantum_circuit

        # Create expanded batch: for each input, we create one shifted + and one shifted -
        # So total shape will be (batch_size * input_dim * 2, input_dim)
        plus_inputs = input_tensor.unsqueeze(1).repeat(1, input_dim, 1)
        minus_inputs = plus_inputs.clone()

        idx = torch.arange(input_dim)
        plus_inputs[:, idx, idx] += shift
        minus_inputs[:, idx, idx] -= shift

        # Reshape into 2D batch (2 * batch_size * input_dim, input_dim)
        thetas_plus = plus_inputs.reshape(-1, input_dim)
        thetas_minus = minus_inputs.reshape(-1, input_dim)

        # Run both batches
        exp_plus = circuit.run(thetas_plus)    # shape: (2 * B * D, L)
        exp_minus = circuit.run(thetas_minus)

        # Compute parameter-shift gradient estimates
        grads = (exp_plus - exp_minus) / (2 * shift)   # shape: (2 * B * D, L)

        # Print the gradients (quantum gradients)
        # print("Quantum gradients (before reshape):", grads.mean(dim=0))  # Print gradients per logit
        # print("Quantum gradients shape:", grads.shape)  # Ensure it's the expected shape

        # Reshape back to (B, D, L)
        gradients = grads.view(batch_size, input_dim, num_logits)

         # Print gradients after reshaping to check the quantum gradient flow per sample
        # print("Quantum gradients (after reshape):", gradients.mean(dim=0))  # Average across batch

        # Chain rule contraction with grad_output
        final_grads = torch.einsum('bij,bj->bi', gradients, grad_output)

        return final_grads, None, None
