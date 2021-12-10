import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.applications.resnet import preprocess_input, ResNet50

from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
from glob import glob
import random
import os

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, GlobalAveragePooling2D, MaxPool2D, BatchNormalization
from tensorflow.keras.models import Model 

import splitfolders
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator



DIR = 'toyota_image_dataset_v2/toyota_cars/' #TODO: change according to 1/3 datasets 
TEST_RATIO = 0.13
FOLDER_CNT = 5
NUM_CLASSES = 38 #TODO: change according to 1/3 datasets 
BATCH_SIZE = 64 
NO_EPOCHS = 5

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
        if curr_count == FOLDER_CNT - 1:
            break
    # now that we have the split files, return as tuple
    return (train_files, test_files)


def get_model_name(model_file_path):
    slash_pos = model_file_path.rfind('/')
    dash_pos = model_file_path.rfind('-')
    return model_file_path[slash_pos + 1 : dash_pos]


def instantiate_model():
    model = Sequential()
    model.add(ResNet50(include_top=False, pooling='max', weights='imagenet'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    # ResNet-50 model is already trained, should not be trained
    model.layers[0].trainable = True
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    return model 


# TODO: Open and store the image inside each tuple
def assign_labels(dataset):
    labels = []
    images = []
    for file_path in dataset:
        # get the label from the image
        car_name = get_model_name(file_path)
        # open the image and convert to numpy array
        img = image.load_img(file_path, target_size = (224,224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        # store the label and image
        labels.append(car_name)
        images.append(img)
    return (images, labels)


def get_data():
    train_set, test_set = split_dataset()
    train_imgs, train_labels = assign_labels(train_set)
    test_imgs, test_labels = assign_labels(test_set)
    return (train_imgs, train_labels, test_imgs, test_labels)


def main():


    
    DIR = './toyota_image_dataset_v2/toyota_cars/'

    num_skipped = 0
    for folder_name in glob(DIR + '*'):
        folder_path = folder_name
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                fobj = open(fpath, "rb")
                is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
            finally:
                fobj.close()

            if not is_jfif:
                num_skipped += 1
                # Delete corrupted image
                os.remove(fpath)

    print("Deleted %d images" % num_skipped)



    input_folder = "./toyota_image_dataset_v2/toyota_cars"
    #data after splitting images to train, test and val folders
    output = "./toyota_image_dataset_v2/toyota_cars_processed" 
    splitfolders.ratio(input_folder, output, seed=43, ratio=(.7,.3))

    img_size = (224,224)
    batch_size = 32

    train_data_path = "toyota_image_dataset_v2/toyota_cars_processed/train"
    test_data_path = "toyota_image_dataset_v2/toyota_cars_processed/test"


    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                validation_split=0.3
                )

    train_ds = train_datagen.flow_from_directory(
                train_data_path,
                target_size=img_size,
                batch_size=batch_size,
                class_mode= 'categorical',
                subset='training')

    test_ds = train_datagen.flow_from_directory(
                test_data_path,
                target_size=img_size,
                batch_size=1,
                class_mode= 'categorical',
                subset='validation')


    base_model = ResNet50(include_top=False, weights='imagenet')

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(38, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
            layer.trainable = False 

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    seqModel = model.fit(train_ds, epochs = 20)
    train_loss = seqModel.history['loss']
    train_accuracy = seqModel.history['accuracy']
    print(train_loss)
    print(train_accuracy)



    loss, accuracy = model.evaluate(test_ds, verbose=1)
    print('Validation loss:', loss)
    print('Validation accuracy:', accuracy)




if __name__ == '__main__':
    main()