# Classical CNN vs. Hybrid-Quantum Neural Network (HQNN) for Plant Leaf Classification

This project explores the integration of quantum computing into a classical deep learning workflow using a **hybrid quantum neural network**. Specifically, it investigates how a quantum layer, implemented with **CUDA-Q by NVIDIA**, affects performance in an image classification task involving plant leave images from a TensorFlow data source. This project expands upon the CUDA-Q HQNN example from NVIDIA at https://nvidia.github.io/cuda-quantum/latest/applications/python/hybrid_quantum_neural_networks.html.

## Objective
Test whether incorporating a quantum layer into a simple convolutional neural network (CNN) improves or hinders classification accuracy on plant leaf images. The model architecture is deliberately kept simple to ensure the learning from the quantum layer is a key factor in the image classification.

##  Model Architecture
- **1-2 convolutional layers** followed by pooling (classical)
- **3 fully connected layers** (classical)
- **1 (optional) quantum layer**

 <img width="954" alt="hqnn" src="https://github.com/user-attachments/assets/1fbdd611-9b17-449f-8ebe-198b4da157af" />


Within the quantum layer, the main function that allows for learning is the quantum kernel (QK). The QK is essentially a 'function' that can handle most of the basic Python logic (e.g., if/for/while/else/+/-/*) that acts upon qubits in a quantum circuit (QC). The QK takes in some set of inputs (in this case, angle rotations of qubits) that are passed to the QC. The final measurement/observation is outside of the QK, residing inside of QuantumFunction.run() where expectation values are calculated with input Hamiltonian(s).

In this project, the QK is being simulated via a traditional CPU/GPU backend, but real quantum computers can be connected too (https://nvidia.github.io/cuda-quantum/latest/using/backends/hardware.html).  

## Project structure
Only 2 functions from this project are needed to prepare data & run the CNN/HQNN:
1) `download_and_prepare_dataset()`: Downloads, prepares, and augments the TensorFlow plant leaves dataset. Stored in `dataset_builder.py`.
2) `build_and_run_nn()`: Loads the augmented dataset and runs a fully classical CNN or hybrid-quantum neural network. Stored in `model_builder.py`.

-`config` - Local storage configurations and runtime variables <br>
-`utils/constants` - Constants for mapping leaf names to integers for CNN/HQNN training <br>
-`utils/data_utils` - Functions that feed into download_and_prepare_dataset() to manage dataset curation and preparation <br>
-`utils/model_utils` - Functions that feed into build_and_run_nn() and houses the core functionality surrounding CNN/HQNN architecture and training algorithm

## Getting Started
1) Download the  the latest CUDA-Q docker image. Follow https://nvidia.github.io/cuda-quantum/latest/using/quick_start.html.
2) `pip install requirements.txt`
3) Downloading the TensorFlow data source & prepare it for use with PyTorch/CUDA-Q with `download_and_prepare_dataset()`
4) Run a classical CNN or HQNN with `build_and_run_nn()`
5) Improving the HQNN - Modify the quantum kernel in `QuantumFunction` to include different qubit counts, qubit encodings (e.g., angle or basis), or entanglements. Modify the measurement Hamiltonians as part of the `quantum_layer_args` dictionary.

## Technologies
- [PyTorch](https://pytorch.org/) - Core neural network model setup.
- [CUDA-Q](https://developer.nvidia.com/cuda-quantum) - Quantum layer implementation compatible with PyTorch.
- [TensorFlow](https://www.tensorflow.org/datasets) - Source of plant leaf images. plant_leaves - 4500 healthy/unhealthy leaves, 22 species/health categories
ImageCLEF 2013 plant task dataset.
- Python 3.8+
- GPU acceleration (CUDA-compatible)
