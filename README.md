# Classical CNN vs. Hybrid-Quantum Neural Network (HQNN) for Plant Leaf Classification

This project explores the integration of quantum computing into a classical deep learning workflow using a **hybrid quantum neural network**. Specifically, it investigates how a quantum layer, implemented with **CUDA-Q by NVIDIA**, affects performance in an image classification task involving plant leave images from a TensorFlow data source.

## Objective
Test whether incorporating a quantum layer into a simple convolutional neural network (CNN) improves or hinders classification accuracy on plant leaf images. The model architecture is deliberately kept simple to ensure the learning from the quantum layer is a key factor in the image classification.

##  Model Architecture
- **1-2 convolutional layers** followed by pooling (classical)
- **2-3 fully connected layers** (classical)
- **1 (optional) quantum layer**

## Project structure
Only 2 functions from this project are needed to prepare data & run the CNN/HQNN:
1) download_and_prepare_dataset(): Downloads, prepares, and augments the TensorFlow plant leaves dataset. Stored in dataset_builder.py.
2) build_and_run_nn(): Loads the augmented dataset and runs a fully classical CNN or hybrid-quantum neural network. Stored in model_builder.py.

Local storage configurations and runtime variables are modified in config.yml.

## Technologies
- [PyTorch](https://pytorch.org/) - Core neural network model setup.
- [CUDA-Q](https://developer.nvidia.com/cuda-quantum) - Quantum layer implementation compatible with PyTorch.
- [TensorFlow](https://www.tensorflow.org/datasets) - Source of plant leaf images.
- Python 3.8+
- GPU acceleration (CUDA-compatible)
