import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.image import resize
from tensorflow.keras.models import load_model

DATA_DIR = "./datasets/ff1010"
classes = [1,0] # 1 = Bird, 0 = No bird

def preprocess_data(data_dir, class_names, target_shape=(128,128)):
    data = []
    labels = []

    label_file = os.path.join(data_dir, 'ff1010bird_metadata_2018.csv')
    audio_folder = os.path.join(data_dir, 'wav')

    csv = pd.read_csv()