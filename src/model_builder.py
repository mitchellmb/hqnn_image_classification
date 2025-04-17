import torch
import cudaq
import time
import pandas as pd
import numpy as np
from typing import Literal
from src.config.config import Config
from src.utils.model_utils.hybrid_nn import HybridNN
from src.utils.model_utils.model_setup import (
    batch_train_model, 
    load_augmented_dataset,
    format_y_labels, 
    get_n_classes, 
    get_nn_shape_after_pooling)


DATA_CONFIG = Config.get('data')
TRAINING_CONFIG = Config.get('training_parameters')


device = torch.device('cuda' if torch.cuda.is_available() and TRAINING_CONFIG.get('cuda_device') == 'gpu' else 'cpu')
cudaq.set_target("qpp-cpu") 


def build_and_run_nn(label_target: Literal['categories', 'binary'], cnn_fc_layer_args: dict, 
                     quantum_layer_args: dict, training_params: dict):
    '''
    Control function for building and running the Hybrid Quantum Neural Network.

    Loads in the augmented images dataset. 
    Formats the targets for CrossEntropyLoss. 
    Initializes the NN with paramters determined by the config.yml and input dimensions for CNN/FC layers.
    Runs the batch training NN algorithm.
    Saves the model & training costs/accuracies for plotting later.
    '''

    # 1 - load augmented data
    BASE_FILE_PATH = DATA_CONFIG.get('base_file_path')
    AUGMENTED_DATA_SAVE_LOC = BASE_FILE_PATH + DATA_CONFIG.get('local_file_augmented')
    _file_name = AUGMENTED_DATA_SAVE_LOC + f'_{label_target}.pkl'
    x_train, x_test, y_train, y_test = load_augmented_dataset(_file_name)

    # 2 - format y labels for CrossEntropyLoss
    y_train, y_test = format_y_labels(y_train, y_test)

    # 3 - input parameter setup for NN class
    n_classes = get_n_classes(y_train)
    print(f'Number of classes {n_classes}')
    print(f'Y values {y_train.min(), y_train.max()}\n')

    shape_after_pooling = get_nn_shape_after_pooling(x_train)

    # 4 - initialize neural network
    nn = HybridNN(
        n_classes=n_classes, 
        shape_after_pooling = shape_after_pooling, 
        conv_channels_1 = cnn_fc_layer_args.get('conv_channels_1'),
        conv_channels_2 = cnn_fc_layer_args.get('conv_channels_2'),
        fc_neuron_ct_1 = cnn_fc_layer_args.get('fc_neurons_1'),
        fc_neuron_ct_2 = cnn_fc_layer_args.get('fc_neurons_2'),
        dropout = cnn_fc_layer_args.get('dropout'),
        quantum_layer_args=quantum_layer_args)
    
    # 5 - run the NN 
    train_cost, train_accuracy, test_cost, test_accuracy, model_out = batch_train_model(
        x_train, x_test, y_train, y_test,
        nn_model = nn.to(device), 
        device = device,
        learning_rate = training_params.get('learning_rate_init'),
        regularization = training_params.get('l2_regularization'),
        batch_size = training_params.get('batch_size'),
        n_epochs = training_params.get('epochs'),
        early_stopping_patience = training_params.get('early_stopping'))
    
    # 6 - save the training outputs
    df = pd.DataFrame(np.transpose([train_cost, train_accuracy, test_cost, test_accuracy]),
                      columns=['training_cost', 'training_accuracy', 'testing_cost', 'testing_accuracy'])
    
    t_out = str(time.time()).split('.')[0]
    if quantum_layer_args:
        file_out = BASE_FILE_PATH + 'hybrid_quantum_nn' + f'_{t_out}'
    else: 
        file_out = BASE_FILE_PATH + 'traditional_nn' + f'_{t_out}'

    df.to_csv(file_out + '.csv', index=False)
    torch.save(model_out.state_dict(), file_out + '.pt')
    print(f'Saved neural network fitting results and model to {file_out}')

    return train_cost, train_accuracy, test_cost, test_accuracy, model_out, file_out

