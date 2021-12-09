from glob import glob
from PIL import Image
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2

# separate into train and test dataset
DIR = './toyota_image_dataset_v2/toyota_cars/'
TEST_RATIO = 0.13

def split_dataset():
    test_files = []
    train_files = []
    # iterate over all folders
    car_directories = glob(DIR + "*/")
    curr_count = 0
    for directory in car_directories:
        files = glob(directory + '*.jpg')
        # pick like 70 images from each folder for test, rest train
        curr_test_set = random.sample(files, int(TEST_RATIO * len(files)))
        curr_train_set = [x for x in files if x not in curr_test_set]
        # add it to the master lists
        test_files.extend(curr_test_set)
        train_files.extend(curr_train_set)
        curr_count += 1
    # now that we have the split files, return as tuple
    return (train_files, test_files)


def get_model_name(model_file_path):
    # master_labels = ['aygo', 'avanza', '4runner', 'avensis', 'alphard', 'camry', 'avalon']
    # master_labels = ['aygo', 'avanza']
    master_labels = ['crown', 'hilux', 'vitz', 'hiace', 'corona', 'aygo', 'avanza', '4runner', 'vios', 'sequoia', 'avensis', 'supra', 'sienna', 'etios', 'prius', 'rush', 'previa', 'innova', 'highlander', 'revo', 'tacoma', 'alphard', 'starlet', 'mirai', 'tundra', 'venza', 'yaris', 'iq', 'estima', 'soarer', 'verso', 'matrix', 'camry', 'avalon', 'rav4', 'celica', 'corolla', 'fortuner']
    slash_pos = model_file_path.rfind('/')
    dash_pos = model_file_path.rfind('-')
    curr_car_name = model_file_path[slash_pos + 1 : dash_pos]
    curr_labels = [0] * len(master_labels)
    index = master_labels.index(curr_car_name)
    curr_labels[index] = 1
    return curr_labels

# TODO: Open and store the image inside each tuple
def assign_labels(dataset):
    labels = []
    images = []
    img_size = (128, 128)
    for file_path in dataset:
        # get the label from the image
        car_name = get_model_name(file_path)
        # open the image and convert to numpy array
        try:
            img = cv2.imread(file_path)
            img_resized = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_CUBIC)
            img_list = img_resized.tolist()
        except:
            continue
        # store the label and image
        labels.append(car_name)
        images.append(img_list)
    labels = tf.convert_to_tensor(labels, dtype=tf.float32)
    images = np.array(images)
    images = (images / 255).astype(np.float32)
    return (images, labels)

def get_data():
    train_set, test_set = split_dataset()
    print('Successfully split dataset')
    train_imgs, train_labels = assign_labels(train_set)
    print('Successfully assigned labels to training set')
    test_imgs, test_labels = assign_labels(test_set)
    print('Successfully assigned labels to testing set')
    print('Preprocessing done')
    return (train_imgs, train_labels, test_imgs, test_labels)
