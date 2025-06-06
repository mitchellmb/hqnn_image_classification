import warnings
import torch
import torch.nn as nn
from src.utils.model_utils.quantum_layer import QuantumLayer


class HybridNN(nn.Module):
    def __init__(
            self,
            n_classes: int,
            shape_after_pooling: int,
            conv_channels_1=None,
            conv_channels_2=None,
            fc_neuron_ct_1=128,
            fc_neuron_ct_2=64,
            input_channels=3, 
            dropout=0.3,
            quantum_layer_args=None):

        super(HybridNN, self).__init__()
        self.quantum = True if quantum_layer_args else False
        self.n_classes = n_classes
        self.channel_out_shape = 1
        self.conv_channels_1 = conv_channels_1
        self.conv_channels_2 = conv_channels_2

        if self.quantum and fc_neuron_ct_2 > 0:
            warnings.warn(f'''Quantum arguments were given alongside fc_neurons_2. 
                          The value of fc_neurons_2 will be ignored as its output
                          must match qubit_count*features_per_qubit for angle
                          encoding.''')

        if self.conv_channels_1:
            self.conv1 = nn.Conv2d(input_channels, self.conv_channels_1, kernel_size=3, padding=1) 
            self.bn1 = nn.BatchNorm2d(self.conv_channels_1)
            self.pool = nn.MaxPool2d(kernel_size=2)
            self.channel_out_shape = self.conv_channels_1

        if self.conv_channels_2:
            self.conv2 = nn.Conv2d(self.conv_channels_1, self.conv_channels_2, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(self.conv_channels_2)
            self.channel_out_shape = self.conv_channels_2

        self.fc1 = nn.Linear(self.channel_out_shape * shape_after_pooling, fc_neuron_ct_1)
        
        if self.quantum: 
            self.quantum_scale = nn.Parameter(torch.tensor(quantum_layer_args.get('quantum_param_init_scale', 1))) 
            self.features_per_qubit = quantum_layer_args.get('features_per_qubit', 1)
            self.qubit_count = quantum_layer_args.get('qubit_count', 5)

            self.fc2 = nn.Linear(fc_neuron_ct_1, self.qubit_count*self.features_per_qubit) 
            self.quantum = QuantumLayer(qubit_count=self.qubit_count, quantum_layer_args=quantum_layer_args)
            self.fc_out = nn.Linear(self.qubit_count, self.n_classes)
        else:
            self.fc2 = nn.Linear(fc_neuron_ct_1, fc_neuron_ct_2) 
            self.fc_out = nn.Linear(fc_neuron_ct_2, self.n_classes)

        self.dropout = nn.Dropout(dropout) 

    def forward(self, x):

        if self.conv_channels_1:
            x = torch.relu(self.conv1(x)) 
            x = self.bn1(x)

        if self.conv_channels_2:
            x = torch.relu(self.conv2(x)) 
            x = self.bn2(x)

        if self.conv_channels_1:
            x = self.pool(x) 

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = self.dropout(x) 
        x = nn.functional.leaky_relu(self.fc2(x), negative_slope=0.1) # allows negative outputs
        x = self.dropout(x)

        if self.quantum:
            x = self.quantum(x)
            x = self.fc_out(x*self.quantum_scale) # amplifies small quantum outputs
        else:
            x = self.fc_out(x) 
            
        return x