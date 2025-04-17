import gc
import os
import pickle
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def download_tf_data_source(dataset_name):
    print('Downloading original TensorFlow image data source...')

    ds_builder = tfds.builder(dataset_name)
    datasource = ds_builder.as_data_source(split='train') #train data sources include labels
    ds_info = ds_builder.info

    n_labels = len(set(ds_info.features['label'].names))
    ds_labels_dict = dict(zip(np.arange(0, n_labels, 1), ds_info.features['label'].names))

    return datasource, ds_labels_dict


def find_key_with_value(d, target):
    for k, v_list in d.items():
        if target in v_list:
            return k
        
    return None


def get_binary_labels(labels_dict):
    binary_labels = {'healthy':[], 'unhealthy':[]}

    for k, v in labels_dict.items():
        if 'healthy' in v:
            binary_labels['healthy'].append(k)
        else:
            binary_labels['unhealthy'].append(k)

    return binary_labels


def write_chunk_to_file(chunk, file_path):
    with open(file_path, 'ab') as f:
        for record in chunk:
            pickle.dump(record, f)


def reduce_image_size(image_in, height, width):
    im_tensor = tf.convert_to_tensor(image_in)
    resize_im = tf.image.resize(im_tensor, (height, width))
    resize_im = np.clip(np.round(resize_im.numpy()), 0, 255).astype(np.uint8)

    return resize_im


def preprocess_chunk(chunk, labels_dict, binary_labels_dict, image_dims):
    #need new data list, tf data source is immutable
    updated_records=[] 
    for record in chunk:
        
        record['label_str'] = labels_dict.get(record['label'])
        record['label_binary'] = find_key_with_value(binary_labels_dict, record['label'])

        record['image_reduced'] = reduce_image_size(record['image'], image_dims[0], image_dims[1])
        record.pop('image')

        updated_records.append(record)

    return updated_records


def rescale_and_save(local_save_loc, tf_data_source, chunk_size, **kwargs):
    if os.path.exists(local_save_loc):
        os.remove(local_save_loc)
        print(f'{local_save_loc} has been deleted. Continuing to overwrite with new preprocessed data source...')
        
    iterator = iter(tf_data_source)

    while True:
        # input image sizes are large from tf data sources, process in smaller chunks
        chunk = []

        try:
            # get up to chunk_size elements from data source
            chunk = [next(iterator) for _ in range(chunk_size)]
        except StopIteration:
            pass

        if not chunk:
            # iterated through all elements in data source
            break

        updated_chunk = preprocess_chunk(chunk, **kwargs)
        write_chunk_to_file(updated_chunk, file_path=local_save_loc)

        del chunk
        del updated_chunk
        gc.collect()

    print(f'TensorFlow data source rescaled and saved to: {local_save_loc}')


def download_and_save_data_source(local_save_loc, tf_data_source_name, image_output_dimensions, chunk_size):
    tf_data_source, labels = download_tf_data_source(tf_data_source_name)
    binary_labels = get_binary_labels(labels)

    rescale_and_save(
        local_save_loc, 
        tf_data_source, 
        chunk_size=chunk_size, 
        labels_dict=labels, 
        binary_labels_dict=binary_labels, 
        image_dims=image_output_dimensions)