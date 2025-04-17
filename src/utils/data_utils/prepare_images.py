import pickle
import numpy as np
import tensorflow as tf
import torch
from torchvision import transforms


def convert_tf_image_to_torch(ndarr_image):
    return torch.from_numpy(ndarr_image).permute(2,0,1).float()


def convert_dataset_to_torch(x_train, x_test, y_train, y_test):
    x_train_tensor = torch.stack([convert_tf_image_to_torch(x) for x in x_train])
    x_test_tensor = torch.stack([convert_tf_image_to_torch(x) for x in x_test])

    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    return x_train_tensor, x_test_tensor, y_train_tensor, y_test_tensor


def mask_grey_background(image, grey_threshold):
    mask_background = np.all(np.abs(image - image.mean(axis=2, keepdims=True)) < grey_threshold, 
                                    axis=2)

    masked_image = image.copy()
    masked_image[mask_background] = 0

    return masked_image


def get_image_mean_std(images):
    image_means = tf.reduce_mean(images, axis=(0,2,3)).numpy() #rgb is idx 1
    image_std = tf.math.reduce_std(images, axis=(0,2,3)).numpy()

    return image_means, image_std


def get_crop_sizes(image, crop_percent_sides=0, crop_percent_top_bottom=0):
    _, width, height = image.shape

    cropped_width = int(width-crop_percent_sides/100.*width)
    cropped_height = int(height-crop_percent_top_bottom/100.*height)

    return cropped_width, cropped_height


def apply_tensor_transformations(images_train, images_test, transformation_pipe):
    train_transformed = torch.stack([transformation_pipe(im) for im in images_train])
    test_transformed = torch.stack([transformation_pipe(im) for im in images_test])

    return train_transformed, test_transformed


def save_augmented_dataset(local_save_loc, x_train, x_test, y_train, y_test):
    combined_data = {
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test}

    with open(local_save_loc, 'wb') as f:
        pickle.dump(combined_data, f)


def augment_dataset(x_train, x_test, y_train, y_test, grey_threshold, crop_side, crop_top_bottom):

    # masks grey background observed in all plant leaf images
    x_train_masked = [mask_grey_background(im, grey_threshold) for im in x_train]
    x_test_masked = [mask_grey_background(im, grey_threshold) for im in x_test]

    # converts tf to torch tensors
    x_train, x_test, y_train, y_test = convert_dataset_to_torch(x_train_masked, 
                                                                x_test_masked, y_train, y_test)

    # image dataset augmentations
    cropped_width_px, cropped_height_px = get_crop_sizes(x_train[0],
                                                         crop_percent_sides=crop_side,
                                                         crop_percent_top_bottom=crop_top_bottom)

    transformations = transforms.Compose([
        transforms.Lambda(lambda x: x/255),
        transforms.CenterCrop((cropped_height_px, cropped_width_px)),
        transforms.RandomRotation(90),
        transforms.RandomHorizontalFlip()])

    x_train, x_test = apply_tensor_transformations(x_train, x_test, transformations)

    return x_train, x_test, y_train, y_test