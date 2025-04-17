import pickle
import torch
import tqdm
import copy
import torch.nn as nn


def load_augmented_dataset(file_name):
    with open(file_name, 'rb') as f:
        loaded_data = pickle.load(f)

    x_train = loaded_data.get('x_train')
    x_test = loaded_data.get('x_test')
    y_train = loaded_data.get('y_train')
    y_test = loaded_data.get('y_test')

    if len(x_train) == 0 or len(x_test) == 0 or len(y_train) == 0 or len(y_test) == 0:
        raise ValueError(f"Dataset is incomplete or empty. Check the source file {file_name}")
    
    return x_train, x_test, y_train, y_test


def format_y_labels(y_train, y_test):
    unique_yvals = torch.unique(torch.cat((y_train, y_test)))
    yval_to_int_mapping = {label.item(): idx for idx, label in enumerate(unique_yvals)}

    y_train_int = torch.tensor([yval_to_int_mapping[label.item()] for label in y_train])
    y_test_int = torch.tensor([yval_to_int_mapping[label.item()] for label in y_test])

    return y_train_int, y_test_int


def get_n_classes(y_values):
    y_values_np = y_values.numpy()
    return len(set(y_values_np))


def get_nn_shape_after_pooling(x_train, n_pool_layers=1, pool_size=2, stride=2):
    input_dimensions = list(x_train.shape)
    height, width = input_dimensions[2], input_dimensions[3]

    for _ in range(n_pool_layers):
        height = (height - pool_size) // stride + 1
        width = (width - pool_size) // stride + 1

    shape_after_pooling = int(height) * int(width)

    return shape_after_pooling


def accuracy_score(y_hat, y):
    _, y_preds = torch.max(y_hat, dim=1)
    correct_preds = (y_preds == y).float()
    return correct_preds.mean()


def batch_train_model(x_train, x_test, y_train, y_test,
                      nn_model, device,
                      learning_rate=0.001, 
                      regularization=1e-3, 
                      batch_size=50, 
                      n_epochs=1000, 
                      early_stopping_patience=10,
                      gradient_clip_value=1.0):

    # initialize
    best_accuracy = -1
    best_params = None
    slice_batch = torch.arange(0, len(x_train), batch_size)
    epochs_without_improvement = 0

    loss_fxn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=learning_rate, weight_decay=regularization)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # train
    training_cost = []
    testing_cost = []
    training_accuracy = []
    testing_accuracy = []

    for i in range(n_epochs):

        print(f'Epoch {i}, Learning Rate: {optimizer.param_groups[0]["lr"]}')

        nn_model.train()
        with tqdm.tqdm(slice_batch, unit='batch', mininterval=0, disable=False) as bar:
            bar.set_description(f'epoch {i}')

            for idx in bar:
                x_batch = x_train[idx:idx+batch_size]
                y_batch = y_train[idx:idx+batch_size]
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                # forward
                y_hat_batch = nn_model(x_batch)
                train_cost = loss_fxn(y_hat_batch, y_batch) 

                # backward
                optimizer.zero_grad()
                train_cost.backward()

                torch.nn.utils.clip_grad_norm_(nn_model.parameters(), gradient_clip_value)
                optimizer.step()

                train_accuracy = accuracy_score(y_hat_batch, y_batch)
                bar.set_postfix(loss=float(train_cost), acc=float(train_accuracy))

        # evaluate per epoch with all test/train data
        nn_model.eval()
        with torch.no_grad():
            y_hat_training_all = nn_model(x_train.to(device))
            train_cost_all = loss_fxn(y_hat_training_all, y_train.to(device))
            train_accuracy_all = accuracy_score(y_hat_training_all, y_train.to(device))

            training_cost.append(train_cost_all)
            training_accuracy.append(train_accuracy_all)

            y_hat_test = nn_model(x_test.to(device))
            test_cost = loss_fxn(y_hat_test, y_test.to(device))
            test_accuracy = accuracy_score(y_hat_test, y_test.to(device)) 

            testing_cost.append(test_cost)
            testing_accuracy.append(test_accuracy)

        print(f"Training accuracy (all samples) = {training_accuracy[-1]:.3f}")
        print(f"Test accuracy = {testing_accuracy[-1]:.3f}\n")

        if training_accuracy[-1] > best_accuracy:
            best_accuracy = training_accuracy[-1]
            best_params = copy.deepcopy(nn_model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if training_accuracy[-1] > 0.995:
            print(f"Early stopping at epoch {i}, training accuracy > 99.5%.")
            break
        elif epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping at epoch {i} due to training accuracy plateau.")
            break   

        nn_model.load_state_dict(best_params)
        scheduler.step()
    
    return training_cost, training_accuracy, testing_cost, testing_accuracy, nn_model



