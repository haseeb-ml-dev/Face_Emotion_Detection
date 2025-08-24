import numpy as np
import pandas as pd
import tensorflow as tf
Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
Dropout = tf.keras.layers.Dropout
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense
to_categorical = tf.keras.utils.to_categorical
Sequential = tf.keras.models.Sequential
from sklearn.model_selection import train_test_split

# 1. Load FER-2013 dataset (CSV format)
# The dataset is usually available on Kaggle
# It has 'emotion' column (labels) and 'pixels' column (48x48 grayscale values)

data = pd.read_csv("fer2013.csv")

# 2. Extract features and labels
pixels = data['pixels'].tolist()
faces = []

for pixel_sequence in pixels:
    face = [int(pixel) for pixel in pixel_sequence.split(' ')]
    face = np.asarray(face).reshape(48, 48)
    faces.append(face)

faces = np.array(faces)
faces = np.expand_dims(faces, -1)  # add channel dimension
faces = faces / 255.0  # normalize

emotions = to_categorical(data['emotion'], num_classes=7)

# 3. Train-test split
x_train, x_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.2, random_state=42)

# 4. Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Train the model
model.fit(x_train, y_train, epochs=25, batch_size=64, validation_data=(x_test, y_test))

# 6. Save the model
model.save("emotion_model.h5")
