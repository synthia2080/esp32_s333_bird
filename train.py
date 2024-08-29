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
import keras_tuner as kt

DATA_DIR = "./datasets/ff1010"
classes = [1,0] # 1 = Bird, 0 = No bird


def relu6(x):
    return min(max(0,x), 6)


class invert_res_block(keras.Model):
    def __init__(self, expand, squeeze, kernel):
        super(invert_res_block, self).__init__()

        self.conv2D_squeeze = layers.Conv2D(expand, (1,1), activation='swish')
        self.bn1 = layers.BatchNormalization()
        self.depthwise_conv2D_33 = layers.DepthwiseConv2D(kernel, activation='swish', padding='same')
        self.bn2 = layers.BatchNormalization()
        self.conv2D_expand = layers.Conv2D(squeeze, (1,1))
        self.bn3 = layers.BatchNormalization()
        
        self.adjust_channels = layers.Conv2D(squeeze, (1, 1), padding='same')

    def call(self, input_tensor):
        x = self.conv2D_squeeze(input_tensor)
        x = self.bn1(x)
        x = self.depthwise_conv2D_33(x)
        x = self.bn2(x)
        x = self.conv2D_expand(x)
        x = self.bn3(x)

        if x.shape[-1] != input_tensor.shape[-1]:
            input_tensor = self.adjust_channels(input_tensor)

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


def build_model(hp):
    model = keras.Sequential()
    hp_units1 = hp.Int("units1", min_value=32, max_value = 512, step=32)
    hp_units2 = hp.Int("units2", min_value=32, max_value = 512, step=32)
    hp_kernels1 = hp.Choice("kernel1", [1,2,3,4,5,6])
    hp_kernels2 = hp.Choice("kernel2", [1,2,3,4,5,6])
    hp_kernel3_MP = hp.Choice("kernel3_MP", [1,2,3,4,5,6])
    hp_kernels4 = hp.Choice("kernel4", [1,2,3,4,5,6])
    # hp_units3_res = hp.Choice("units3_res", [1,2,3,4,5,6])
    hp_units4_res = hp.Choice("units4_res", [1,2,3,4,5,6])

    model.add(layers.Conv2D(hp_units1, hp_kernels4, activation='relu'))
    model.add(layers.Conv2D(hp_units2, hp_kernels4, activation='relu'))
    model.add(layers.MaxPool2D(hp_kernel3_MP))
    model.add(invert_res_block(hp_units2, hp_units4_res, hp_kernels1))
    model.add(invert_res_block(hp_units2, hp_units4_res, hp_kernels2))
    model.add(invert_res_block(hp_units2, hp_units4_res, hp_kernels1))
    model.add(layers.Conv2D(hp_units1, hp_kernels4, activation='relu'))
    model.add(layers.Conv2D(hp_units2, hp_kernels4, activation='relu'))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(len(classes), activation='softmax'))

    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model




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
    print(input_shape)

    model = keras.Sequential(
        [
        layers.Conv2D(32, (2,2), activation='relu'),
        layers.Conv2D(128, (2,2), activation='relu'),
        layers.MaxPool2D((2,2)),
        invert_res_block(128, 16, (5,5)),
        invert_res_block(128, 16, (1,1)),
        invert_res_block(128, 16, (5,5)),
        layers.Conv2D(128, (2,2), activation='relu'),
        layers.Conv2D(32, (2,2), activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Flatten(),
        layers.Dense(len(classes), activation='softmax')
        ]
    )

    # try:
    #     tuner = kt.Hyperband(build_model, objective='val_accuracy', max_epochs=10, factor=3, directory='./tuning', project_name='test_1')
    #     tuner.search(x_train,y_train,epochs=10, validation_split=0.2)
    # except Exception as e:
    #     print(f"Error occured during tuning: {e}")

    # best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # print(f"""
    #     Hyperparameter search is complete. The best parameters are:\n
    #     units1: {best_hps.get('units1')}\n
    #     units2: {best_hps.get('units2')}\n
    #     units4_res {best_hps.get('units4_res')}\n
    #     kernel1: {best_hps.get('kernel1')}\n
    #     kernel2: {best_hps.get('kernel2')}\n
    #     kernel3_MP: {best_hps.get('kernel3_MP')}\n
    #     kernel4: {best_hps.get('kernel4')}\n
    #        """)

    
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=15, batch_size=4, validation_data=(x_test,y_test))
    model.save('./saved_models/model_v0.H5')

    
    test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    print(test_accuracy[1])


if __name__ == "__main__":
    main()