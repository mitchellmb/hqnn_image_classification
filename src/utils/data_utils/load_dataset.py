import pickle
from typing import Literal
from sklearn.model_selection import train_test_split
from src.utils.constants import combine_healthy_unhealthy_dict, leaves_target_dict


def load_chunk_from_pickle(file_name):
    with open(file_name, 'rb') as f:  
        chunk = []
        try:
            while True:
                chunk.append(pickle.load(f)) 
        except EOFError:
            pass

    return chunk


def combine_diseased_and_healthy(y_values):
    y_updated=[]
    for y in y_values:
        y_int = int(y)
        y_updated.append(combine_healthy_unhealthy_dict.get(y_int))

    return y_updated


def remove_unpaired_leaves(list1, list2):
    # removes leaves that do not have a healthy/unhealthy counterpart
    idx_to_remove = [i for i, v in enumerate(list2) if v == -1]

    for i in reversed(idx_to_remove):
        del list1[i]
        del list2[i]
    
    return list1, list2


def get_labels_by_leaf_names(leaf_names):
    leaves_target_inverse = {v:k for k, v in leaves_target_dict.items()}

    healthy_and_unhealthy_labels=[]
    for leaf in leaf_names:
        labels = [v for k, v in leaves_target_inverse.items() if leaf.lower() in k.lower()]
        healthy_and_unhealthy_labels.extend(labels)

    return healthy_and_unhealthy_labels


def load_leaves_dataset(file_loc, test_split_size, target: Literal['categories', 'binary'], leaves_subset_list=None): 
    data = load_chunk_from_pickle(file_loc)
    images = [i.get('image_reduced') for i in data]

    # binary - healthy vs. unhealthy leaves
    if target == 'binary':
        targets_str = [i.get('label_binary') for i in data]
        targets = [1 if i == 'healthy' else 0 for i in targets_str] 
        return train_test_split(images, targets, test_size=test_split_size, shuffle=True, random_state=0)
    
    # categories - labeled leaf names, combines healthy & unhealthy pairs
    targets = [i.get('label') for i in data]

    if leaves_subset_list: 
        leaves_labels = get_labels_by_leaf_names(leaves_subset_list)
        targets_idx = [idx for idx, val in enumerate(targets) if val in leaves_labels]
        targets = [targets[i] for i in targets_idx]
        images = [images[i] for i in targets_idx]

    _x_train, _x_test, _y_train, _y_test = train_test_split(images, targets, 
                                                            test_size=test_split_size, 
                                                            shuffle=True, random_state=0)

    _y_train_comb = combine_diseased_and_healthy(_y_train)
    _y_test_comb = combine_diseased_and_healthy(_y_test)

    x_train, y_train = remove_unpaired_leaves(_x_train, _y_train_comb)
    x_test, y_test = remove_unpaired_leaves(_x_test, _y_test_comb)

    return x_train, x_test, y_train, y_test
        

    


