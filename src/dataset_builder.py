import os
from typing import Literal
from src.config.config import Config
from src.utils.data_utils.download_dataset import download_and_save_data_source
from src.utils.data_utils.load_dataset import load_leaves_dataset
from src.utils.data_utils.prepare_images import augment_dataset, save_augmented_dataset


DATA_CONFIG = Config.get('data')
TRAINING_CONFIG = Config.get('training_parameters')


def download_and_prepare_dataset(label_target: Literal['categories', 'binary'], download_data_source: bool):
        '''
        Main control function to prepare a tensorflow images dataset for use with pytorch neural networks.

        Downloads the raw dataset and then filters, resizes, augments, train/test splits, 
        and saves the image dataset. Parameters are determined in config.yml.
        '''

        # 1 - Initial data download or rewrite to new image dimensions set in .yml
        data_source_save_loc = DATA_CONFIG.get('base_file_path') + DATA_CONFIG.get('local_file') + '.pkl'

        if download_data_source: 
                download_and_save_data_source(
                        data_source_save_loc, 
                        tf_data_source_name=DATA_CONFIG.get('dataset'), 
                        image_output_dimensions=[DATA_CONFIG.get('image_height'), 
                                                 DATA_CONFIG.get('image_width')], 
                        chunk_size=DATA_CONFIG.get('chunking_size'))
        else:
                if not os.path.exists(data_source_save_loc):
                        raise FileNotFoundError(f'''{data_source_save_loc} does not already exist. 
                                Check the PATH or set download_data_source = True.''')


        # 2 - Train/test split of leaf data subset
        x_train_np, x_test_np, y_train_np, y_test_np = load_leaves_dataset(
                data_source_save_loc,
                test_split_size=TRAINING_CONFIG.get('train_test_split'),
                target=label_target,
                leaves_subset_list=TRAINING_CONFIG.get('leaves').split(', '))

        print(f'Training dataset length: {len(x_train_np)}')
        print(f'Testing dataset length: {len(x_test_np)}')


        # 3 - Augment subset data for use in neural network training
        augmented_file_name = (DATA_CONFIG.get('base_file_path') + 
                               DATA_CONFIG.get('local_file_augmented') + 
                               f'_{label_target}.pkl')

        x_train, x_test, y_train, y_test = augment_dataset(
                x_train_np, x_test_np, y_train_np, y_test_np, 
                grey_threshold=DATA_CONFIG.get('grey_threshold'),
                crop_side=DATA_CONFIG.get('crop_side'),
                crop_top_bottom=DATA_CONFIG.get('crop_top_bottom'))
        
        save_augmented_dataset(augmented_file_name, x_train, x_test, y_train, y_test)

        print(f'Augmented image dataset saved to {augmented_file_name}')