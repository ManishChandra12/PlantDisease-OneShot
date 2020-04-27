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


def main(cropped):
    if cropped:
        json_file = open('models/cropped_siamese/model_siamese.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("models/cropped_siamese/model.h5")
        loaded_model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(learning_rate=0.001), metrics=['acc'])
    else:
        json_file = open('models/uncropped_siamese/model_siamese.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("models/uncropped_siamese/model.h5")
        loaded_model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(learning_rate=0.001), metrics=['acc'])
    print("Model loaded")
    base_model = loaded_model.get_layer('shared')
    input_shape = (100, 100, 3)
    iinput = Input(shape=input_shape)
    x = base_model(iinput)
    model = Model(iinput, x)
    print(model.summary())

    datagen = ImageDataGenerator(rescale=1. / 255)
    if cropped:
        generator = datagen.flow_from_directory(
            os.path.join(PROCESSED_DATA_DIR, 'PlantDoc-Dataset/train/'),
            target_size=(100, 100),
            batch_size=73,
            class_mode='categorical',
            shuffle=False)
        a = model.predict_generator(generator, 21)
        generator.reset()
        with open('embeddings/cropped_siamese_train_feature.pkl', 'wb') as f:
            pickle.dump(a, f)
        with open('embeddings/uncropped_siamese_train_label.pkl', 'wb') as f:
            pickle.dump(generator.classes, f)

        generator = datagen.flow_from_directory(
            os.path.join(PROCESSED_DATA_DIR, 'PlantDoc-Dataset/test/'),
            target_size=(100, 100),
            batch_size=107,
            class_mode='categorical',
            shuffle=False)
        a = model.predict_generator(generator, 5)
        generator.reset()
        with open('embeddings/cropped_siamese_test_feature.pkl', 'wb') as f:
            pickle.dump(a, f)
        with open('embeddings/uncropped_siamese_test_label.pkl', 'wb') as f:
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
        with open('embeddings/uncropped_siamese_train_feature.pkl', 'wb') as f:
            pickle.dump(a, f)
        with open('embeddings/uncropped_siamese_train_label.pkl', 'wb') as f:
            pickle.dump(generator.classes, f)

        generator = datagen.flow_from_directory(
            os.path.join(RAW_DATA_DIR, 'test/'),
            target_size=(100, 100),
            batch_size=107,
            class_mode='categorical',
            shuffle=False)
        a = model.predict_generator(generator, 5)
        generator.reset()
        with open('embeddings/uncropped_siamese_test_feature.pkl', 'wb') as f:
            pickle.dump(a, f)
        with open('embeddings/uncropped_siamese_test_label.pkl', 'wb') as f:
            pickle.dump(generator.classes, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cropped", action='store_true', help="whether to train on cropped dataset")
    args = parser.parse_args()

    main(args.cropped)
