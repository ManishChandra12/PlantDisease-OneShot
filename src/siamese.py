import os
import argparse
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.applications import InceptionResNetV2
from keras.callbacks import EarlyStopping
from keras import layers
from keras import models
from keras import Input
from keras import Model
from keras import optimizers
import pandas as pd

tf.random.set_seed(1)


def get_flow_from_dataframe(generator, dataframe, image_shape=(100, 100), batch_size=32):
    train_generator_1 = generator.flow_from_dataframe(dataframe, target_size=image_shape,
                                                      x_col='filename1',
                                                      y_col='class',
                                                      class_mode='binary',
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      seed=42)

    train_generator_2 = generator.flow_from_dataframe(dataframe, target_size=image_shape,
                                                      x_col='filename2',
                                                      y_col='class',
                                                      class_mode='binary',
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      seed=42)
    while True:
        x_1 = train_generator_1.next()
        x_2 = train_generator_2.next()

        yield [x_1[0], x_2[0]], x_1[1]


def get_siamese_model(input_shape):
    # Define the tensors for the two input images
    left_input = Input(shape=input_shape)
    right_input = Input(shape=input_shape)

    # Convolutional Neural Network
    conv_base = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
    model = models.Sequential(name='shared')
    model.add(conv_base)
    model.add(layers.Flatten())

    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = layers.Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = layers.Dense(1, activation='sigmoid', kernel_initializer='orthogonal', bias_initializer='zeros')(
        L1_distance)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    # return the model
    return siamese_net


def freeze_layers(model):
    for i in model.layers:
        i.trainable = False
        if isinstance(i, Model):
            freeze_layers(i)
    return model


def main(cropped):
    BATCH_SIZE = 128
    datagen_args = dict(rescale=1. / 255,
                        rotation_range=40,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.2,
                        zoom_range=0.2,
                        horizontal_flip=True)

    train_datagen = ImageDataGenerator(**datagen_args)
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    if cropped:
        df_train = pd.read_csv('train_siamese_data_cropped.csv')
        df_train['class'] = df_train['class'].astype('str')
        df_val = pd.read_csv('val_siamese_data_cropped.csv')
        df_val['class'] = df_val['class'].astype('str')
    else:
        df_train = pd.read_csv('train_siamese_data.csv')
        df_train['class'] = df_train['class'].astype('str')
        df_val = pd.read_csv('val_siamese_data.csv')
        df_val['class'] = df_val['class'].astype('str')

    train_gen = get_flow_from_dataframe(train_datagen, df_train, image_shape=(100, 100),
                                        batch_size=BATCH_SIZE)
    val_gen = get_flow_from_dataframe(val_datagen, df_val, image_shape=(100, 100),
                                      batch_size=BATCH_SIZE)

    model = get_siamese_model((100, 100, 3))
    print(model.summary())

    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(learning_rate=0.001),
                  metrics=['acc'])
    keras_callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, mode='min', min_delta=0.0001, restore_best_weights=True)
    ]

    train_stpep = train_gen.n // train_gen.batch_size
    val_stpep = val_gen.n // val_gen.batch_size
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=train_stpep,
                                  epochs=30,
                                  validation_data=val_gen,
                                  validation_steps=val_stpep,
                                  callbacks=keras_callbacks,
                                  )

    model = freeze_layers(model)
    # serialize model to JSON
    model_json = model.to_json()
    if args.cropped:
        folder = 'models/cropped_siamese'
    else:
        folder = 'models/uncropped_siamese'
    if not os.path.exists(folder):
        os.mkdir(folder)
    with open(folder + "/model_siamese.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(folder + "/model.h5")
    print("Saved model to disk")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cropped", action='store_true', help="whether to train on cropped dataset")
    args = parser.parse_args()

    main(args.cropped)
