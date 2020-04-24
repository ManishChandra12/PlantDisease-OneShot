import argparse
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import InceptionResNetV2
from keras import layers
from keras import models
from keras import optimizers
from keras.callbacks import EarlyStopping
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import os
from CONSTANTS import RAW_DATA_DIR, PROCESSED_DATA_DIR

np.random.seed(1)
tf.random.set_seed(1)


def get_generators(cropped):
    # train data generator
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # test data generator
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    if cropped:
        train_generator = train_datagen.flow_from_directory(
            os.path.join(PROCESSED_DATA_DIR, 'PlantDoc-Dataset/train/'),
            target_size=(100, 100),
            batch_size=32,
            class_mode='categorical')

        test_generator = test_datagen.flow_from_directory(
            os.path.join(PROCESSED_DATA_DIR, 'PlantDoc-Dataset/test/'),
            target_size=(100, 100),
            batch_size=32,
            class_mode='categorical',
            shuffle=False)

        val_generator = test_datagen.flow_from_directory(
            os.path.join(PROCESSED_DATA_DIR, 'PlantDoc-Dataset/val/'),
            target_size=(100, 100),
            batch_size=32,
            class_mode='categorical')
    else:
        train_generator = train_datagen.flow_from_directory(
            os.path.join(RAW_DATA_DIR, 'train/'),
            target_size=(100, 100),
            batch_size=32,
            class_mode='categorical')

        test_generator = test_datagen.flow_from_directory(
            os.path.join(RAW_DATA_DIR, 'test/'),
            target_size=(100, 100),
            batch_size=32,
            class_mode='categorical',
            shuffle=False)

        val_generator = test_datagen.flow_from_directory(
            os.path.join(RAW_DATA_DIR, 'val/'),
            target_size=(100, 100),
            batch_size=32,
            class_mode='categorical')

    return train_generator, val_generator, test_generator


def acc_and_f1(model, generator):
    generator.reset()
    step_size_test = np.ceil(generator.n / generator.batch_size)
    generator.reset()
    y_pred = model.predict_generator(generator, step_size_test)
    idx = np.argmax(y_pred, axis=-1)
    a = np.zeros(y_pred.shape)
    a[np.arange(a.shape[0]), idx] = 1

    generator.reset()
    y_true = np.array(pd.get_dummies(pd.Series(generator.classes)))

    return accuracy_score(y_true, a), f1_score(y_true, a, average='micro')


def main(cropped):
    train_generator, val_generator, test_generator = get_generators(cropped)

    conv_base = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu', kernel_initializer='orthogonal', bias_initializer='zeros'))
    model.add(layers.Dense(27, activation='softmax', kernel_initializer='orthogonal', bias_initializer='zeros'))
    print(model.summary())

    model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(learning_rate=0.001, momentum=0.9),
                  metrics=['acc'])

    keras_callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, mode='min', min_delta=0.0001)
    ]

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=5312 // 32,
                                  epochs=30,
                                  validation_data=val_generator,
                                  validation_steps=1763 // 32,
                                  callbacks=keras_callbacks,
                                  )

    train_generator.reset()
    val_generator.reset()
    train_acc, train_f1 = acc_and_f1(model, train_generator)
    val_acc, val_f1 = acc_and_f1(model, val_generator)
    test_acc, test_f1 = acc_and_f1(model, test_generator)

    print("Final Train Accuracy: " + str(train_acc))
    print("Final Train F1: " + str(train_f1))
    print("Final Val Accuracy: " + str(val_acc))
    print("Final Val F1: " + str(val_f1))
    print("Final Test Accuracy: " + str(test_acc))
    print("Final Test F1: " + str(test_f1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cropped", action='store_true', help="whether to train on cropped dataset")
    args = parser.parse_args()

    main(args.cropped)
