#run pip install split_folders
import splitfolders
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, GlobalAveragePooling2D, MaxPool2D, BatchNormalization
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
#from tensorflow.keras.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

def split_data():
    import os
    from glob import glob
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
    #raw data
    input_folder = "toyota_image_dataset_v2/toyota_cars"
    #where you wanna put data after splitting images to train, test and val folders
    output = "toyota_image_dataset_v2" 
    splitfolders.ratio(input_folder, output, seed=43, ratio=(.6, .2, .2))

def main():

#     split_data()
    img_size = (224,224)
    batch_size = 32
    num_classes = 38

    train_data_path = "toyota_image_dataset_v2/train"
    val_data_path = "toyota_image_dataset_v2/val"
    test_data_path = "toyota_image_dataset_v2/test"


#     ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(
#             rescale=1./255,
#             validation_split=0.4)
        
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                validation_split=0.4)

    train_ds = train_datagen.flow_from_directory(
                train_data_path,
                target_size=img_size,
                batch_size=batch_size,
                class_mode= 'categorical',
                subset='training')

    val_ds = train_datagen.flow_from_directory(
                val_data_path,
                target_size=img_size,
                batch_size=batch_size,
                class_mode= 'categorical',
                subset='validation')

    test_ds = train_datagen.flow_from_directory(
                test_data_path,
                target_size=img_size,
                batch_size=1,
                class_mode= 'categorical',
                subset='validation')

    base_model =tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
            layer.trainable = False

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, epochs = 10)

#     base_model =tf.keras.applications.resnet50.ResNet50(include_top=False, pooling='max', weights='imagenet')
#     x = Dense(num_classes, activation='softmax')
#     model = tf.keras.Sequential()
#     model.add(base_model)
#     model.add(x)
#     model.layers[0].trainable = False
#     model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
#                 loss=tf.keras.losses.BinaryCrossentropy(
#                 from_logits=True, label_smoothing=0.1),
#                 metrics=['accuracy'])
#     model = model.fit(train_ds, validation_data=val_ds, epochs=10,validation_steps=10)


    loss, accuracy = model.evaluate(val_ds, verbose=1)
    print('Validation loss:', loss)
    print('Validation accuracy:', accuracy)


if __name__ == '__main__':
    main()
