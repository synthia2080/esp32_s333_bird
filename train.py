import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import keras
from keras import layers, models, optimizers, utils
from tensorflow import image
import tensorflow as tf
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


class invert_res_block(keras.Model):
    def __init__(self, expand, squeeze, kernel):
        super(invert_res_block, self).__init__()

        self.conv2D_squeeze = layers.Conv2D(squeeze, (1,1), activation='relu')
        self.depthwise_conv2D_33 = layers.DepthwiseConv2D(kernel, activation='relu', padding='same')
        self.conv2D_expand = layers.Conv2D(expand, (1,1), activation='relu')


    def call(self, input_tensor):
        x = self.conv2D_squeeze(input_tensor)
        x = self.depthwise_conv2D_33(x)
        x = self.conv2D_expand(x)

        return tf.math.add(x, input_tensor)

def select_pixels(spectrogram):
    row_median = np.median(spectrogram, axis=0)
    col_median = np.median(spectrogram, axis=1)


    for i, row in enumerate(spectrogram, start=0):
        for i2, item in enumerate(row, start=0):
            if not (item >= row_median[i2]*3 and item >= col_median[i]*3):
                spectrogram[i][i2] = 0

    return spectrogram


def erosion(spectrogram):
    kernel = (3,3)

    blurred = cv2.blur(spectrogram, kernel)
    eroded = cv2.erode(blurred, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)

    return dilated


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
    len_audiopaths = len(audio_paths)
    for i, path in enumerate(audio_paths):

        # Only here for sanity since this shit takes a long time
        if i % 50 == 0:
            print(f"{i}/{len_audiopaths} spectrograms")

        audio_data, sample_rate = librosa.load(path, sr=None)
        spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)

        # Apply post processing 
        spectrogram = select_pixels(spectrogram)
        # spectrogram = erosion(spectrogram)

        spectrogram = image.resize(np.expand_dims(spectrogram, axis=-1), target_shape)
        
        data.append(spectrogram)


    return np.array(labels), np.array(data)



def main():
    labels_out_dir = os.path.join(DATA_DIR, "labels.npy")
    data_out_dir = os.path.join(DATA_DIR, "data.npy")
    # print(f"Built with cuda: {tf.test.is_built_with_cuda()}")
    # print(f"GPU available: {tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)}")
    # print(f"GPU: {print(tf.config.list_physical_devices('GPU'))}")


    # print("Creating Data...")
    # labels, data = preprocess_data(DATA_DIR, (256,256))

    # # Save numpy arrays to save time for late runs
    # np.save(labels_out_dir, labels)
    # np.save(data_out_dir, data)

    labels = np.load(labels_out_dir)
    data = np.load(data_out_dir)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=333)


    input_shape = x_train[0].shape

    model = keras.Sequential(
        [
        layers.Input(shape=input_shape),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPool2D((2,2)),
        invert_res_block(128, 16, (3,3)),
        invert_res_block(128, 16, (3,3)),
        invert_res_block(128, 16, (3,3)),
        layers.GlobalAveragePooling2D(),
        layers.Flatten(),
        layers.Dense(len(classes), activation='softmax')
        ]
    )

    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=50, batch_size=4, validation_data=(x_test,y_test))

    
    test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    print(test_accuracy[1])


if __name__ == "__main__":
    main()