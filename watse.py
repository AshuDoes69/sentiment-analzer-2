import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import nltk
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

nltk.download('stopwords')

# Loading Data
def load_dataset(filepath):
    data = []
    with open(filepath) as f:
        lines = f.readlines()
        for line in lines:
            data.append(line.strip().split(";"))
    return pd.DataFrame(data, columns=["text", "emotion"])

train_data = load_dataset("train.txt")

# Split data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_data["text"], train_data["emotion"], test_size=0.2, random_state=42)

# Text Preprocessing
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)
emotions = le.classes_

tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
max_len = max(len(seq) for seq in X_train_seq)

X_train_padded = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

# Modeling (LSTM)
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=32, input_length=max_len))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(emotions), activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train_padded, y_train_encoded,batch_size=1500 ,epochs=30)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test_padded, y_test_encoded)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Now you can use the model for user input prediction
user_input = input("Enter a text: ")
user_input_seq = tokenizer.texts_to_sequences([user_input])
user_input_padded = pad_sequences(user_input_seq, maxlen=max_len, padding='post')

predicted_emotion_encoded = model.predict(user_input_padded)
predicted_emotion = le.inverse_transform([np.argmax(predicted_emotion_encoded)])
print(f"Predicted Emotion: {predicted_emotion[0]}")
