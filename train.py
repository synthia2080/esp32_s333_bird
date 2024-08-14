import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers, models, optimizers, utils
from tensorflow import image
from sklearn.model_selection import train_test_split
# from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
# from tensorflow.image import resize
# from tensorflow.keras.models import load_model

DATA_DIR = "./datasets/ff1010"
classes = [1,0] # 1 = Bird, 0 = No bird


def preprocess_data(data_dir, target_shape=(128,128)):
    data = []

    label_file = os.path.join(data_dir, 'ff1010bird_metadata_2018.csv')
    audio_folder = os.path.join(data_dir, 'wav')

    df = pd.read_csv(label_file)

    # Add new col with path to wav
    df["wav_path"] = df["itemid"].apply(lambda id: os.path.join(audio_folder, f"{id}.wav"))

    labels = df["hasbird"].tolist()
    audio_paths = df["wav_path"].tolist()

    # Generate spectrograms 
    for path in audio_paths:
        audio_data, sample_rate = librosa.load(path, sr=None)
        spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
        spectrogram = image.resize(np.expand_dims(spectrogram, axis=-1), target_shape)
        
        data.append(spectrogram)

    return np.array(labels), np.array(data)


labels, data = preprocess_data(DATA_DIR)


x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=333)

input_shape = x_train[0].shape
model = keras.Sequential(
    [
    layers.Input(shape=input_shape),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(64,(3,3), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(classes), activation='softmax')
    ]
)

model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test,y_test))

test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(test_accuracy)