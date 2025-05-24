import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

DATA_DIR = "landmark_sequences/all_cut"
LABELS_CSV = "dataset_labels.csv"

df = pd.read_csv(LABELS_CSV)

X, y = [], []
for i, row in df.iterrows():
    data_path = os.path.join(DATA_DIR, row['filename'])
    landmarks = np.load(data_path) 
    if landmarks.shape == (20, 63):  
        X.append(landmarks)
        y.append(row['label'])

X = np.array(X)
y = to_categorical(y) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(20, 63)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(y.shape[1], activation='softmax')  
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

model.save("sign_lstm_model.h5")
print("✅ Model saved as sign_lstm_model.h5")
