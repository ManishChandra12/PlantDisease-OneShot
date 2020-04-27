import os
import argparse
import pickle
import random
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import Model
from keras import Input
from keras.models import model_from_json
from CONSTANTS import RAW_DATA_DIR, PROCESSED_DATA_DIR

random.seed(1)


def main(cropped, which_model):
    folder = None
    file = None
    if which_model == 'resnet':
        if cropped:
            file = 'models/cropped_resnet/model_resnet.json'
            folder = 'models/cropped_resnet/'
        else:
            file = 'models/uncropped_resnet/model_resnet.json'
            folder = 'models/uncropped_resnet/'
    elif which_model == 'siamese':
        if cropped:
            file = 'models/cropped_siamese/model_siamese.json'
            folder = 'models/cropped_siamese/'
        else:
            file = 'models/uncropped_siamese/model_siamese.json'
            folder = 'models/uncropped_siamese/'

    json_file = open(file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(folder + "model.h5")
    if which_model == 'resnet':
        loaded_model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(learning_rate=0.001, momentum=0.9), metrics=['acc'])
    elif which_model == 'siamese':
        loaded_model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(learning_rate=0.001), metrics=['acc'])
    print("Model loaded")

    if which_model == 'siamese':
        base_model = loaded_model.get_layer('shared')
        input_shape = (100, 100, 3)
        iinput = Input(shape=input_shape)
        x = base_model(iinput)
        model = Model(iinput, x)
        print(model.summary())
    elif which_model == 'resnet':
        model = Model(inputs=loaded_model.input, outputs=loaded_model.get_layer('flatten_1').output)
        print(model.summary())

    datagen = ImageDataGenerator(rescale=1. / 255)
    if cropped:
        generator = datagen.flow_from_directory(
            os.path.join(PROCESSED_DATA_DIR, 'PlantDoc-Dataset/train/'),
            target_size=(100, 100),
            batch_size=166,
            class_mode='categorical',
            shuffle=False)
        a = model.predict_generator(generator, 32)
        generator.reset()
        with open('embeddings/cropped_' + which_model + '_train_feature.pkl', 'wb') as f:
            pickle.dump(a, f)
        with open('embeddings/cropped_' + which_model + '_train_label.pkl', 'wb') as f:
            pickle.dump(generator.classes, f)

        generator = datagen.flow_from_directory(
            os.path.join(PROCESSED_DATA_DIR, 'PlantDoc-Dataset/test/'),
            target_size=(100, 100),
            batch_size=257,
            class_mode='categorical',
            shuffle=False)
        a = model.predict_generator(generator, 7)
        generator.reset()
        with open('embeddings/cropped_' + which_model + '_test_feature.pkl', 'wb') as f:
            pickle.dump(a, f)
        with open('embeddings/cropped_' + which_model + '_test_label.pkl', 'wb') as f:
            pickle.dump(generator.classes, f)
    else:
        generator = datagen.flow_from_directory(
            os.path.join(RAW_DATA_DIR, 'train/'),
            target_size=(100, 100),
            batch_size=73,
            class_mode='categorical',
            shuffle=False)
        a = model.predict_generator(generator, 21)
        generator.reset()
        with open('embeddings/uncropped_' + which_model + '_train_feature.pkl', 'wb') as f:
            pickle.dump(a, f)
        with open('embeddings/uncropped_' + which_model + '_train_label.pkl', 'wb') as f:
            pickle.dump(generator.classes, f)

        generator = datagen.flow_from_directory(
            os.path.join(RAW_DATA_DIR, 'test/'),
            target_size=(100, 100),
            batch_size=107,
            class_mode='categorical',
            shuffle=False)
        a = model.predict_generator(generator, 5)
        generator.reset()
        with open('embeddings/uncropped_' + which_model + '_test_feature.pkl', 'wb') as f:
            pickle.dump(a, f)
        with open('embeddings/uncropped_' + which_model + '_test_label.pkl', 'wb') as f:
            pickle.dump(generator.classes, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cropped", action='store_true', help="whether to use cropped dataset")
    parser.add_argument("--model", type=str, choices=['resnet', 'siamese'], default='resnet', help="which trained model to use")
    args = parser.parse_args()

    main(args.cropped, args.model)
