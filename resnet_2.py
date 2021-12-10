#run pip install split_folders

import splitfolder
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, GlobalAveragePooling2D, MaxPool2D, BatchNormalization
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.python.keras.models import Sequential
import numpy as np



def main():
     #raw data
    input_folder = "car_model/input_data"
    #data after splitting images to train, test and val folders
    output = "car_model/processed_data" 
    splitfolder.ratio(input_folder, output, seed=43, ratio=(.6, .2, .2))

    img_size = (224,224)
    batch_size = 32

    train_data_path = "car_model/processed_data/train"
    val_data_path = "car_model/processed_data/val"
    test_data_path = "car_model/processed_data/test"


    ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=0.4)
        
    train_datagen = ImageDataGenerator(preprocessinh_function=preprocess_input,
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


    # x,y = test_ds.next()
    # x.shape

    # base_model =ResNet50(include_top=False, pooling='max', weights='imagenet')
    base_model =ResNet50(include_top=False, weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(train_ds.num_classes, activiation='softmax')(x)
    model = Model(inputs=base_model.input, output=predictions)

    for layer in base_model.layers:
            layer.trainable = True

    model.compile(optimizer='adam', loss='categorical_corssentropy', metrics=['accuracy'])
    model.fit(train_ds, epochs = 10)
    # model.fit(train_ds, validation_data=val_ds, epochs=10, validation_steps=10)

    loss, accuracy = model.evaluate(val_ds, verbose=1)
    print('Validation loss:', loss)
    print('Validation accuracy:', accuracy)


if __name__ == '__main__':
    main()
