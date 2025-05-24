import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

MAX_SEQ_LEN = 10 
FEATURE_VECTOR_SIZE = 256 
NUM_WORD_CLASSES = None  

cnn_model = load_model("sign_model_v2.h5")

feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-3].output)

print("Feature extractor model summary:")
feature_extractor.summary()

sequences = np.load("sequences.npy", allow_pickle=True)

with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)

print(f"Loaded {len(sequences)} sequences and {len(labels)} labels")

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)
NUM_WORD_CLASSES = len(label_encoder.classes_)
print(f"Number of word classes: {NUM_WORD_CLASSES}")

y_categorical = to_categorical(y_encoded, NUM_WORD_CLASSES)


def extract_features(sequences, batch_size=64):
    num_samples, seq_len = sequences.shape[:2]
    features = np.zeros((num_samples, seq_len, FEATURE_VECTOR_SIZE))
    for i in range(num_samples):
        seq = sequences[i]
        feat_seq = feature_extractor.predict(seq, verbose=0)
        features[i] = feat_seq
        if i % 100 == 0:
            print(f"Extracted features for {i}/{num_samples} sequences")
    return features

X_features = extract_features(sequences)

X_padded = pad_sequences(X_features, maxlen=MAX_SEQ_LEN, dtype='float32', padding='post', truncating='post')

X_train, X_test, y_train, y_test = train_test_split(X_padded, y_categorical, test_size=0.2, random_state=42)

model = Sequential([
    LSTM(128, return_sequences=False, input_shape=(MAX_SEQ_LEN, FEATURE_VECTOR_SIZE)),
    Dense(64, activation='relu'),
    Dense(NUM_WORD_CLASSES, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

loss, acc = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f} | Test Accuracy: {acc:.4f}")

model.save("word_lstm_model.h5")

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("Word-level LSTM model and label encoder saved.")
