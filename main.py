import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import nltk
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Additional imports for data cleaning and preprocessing
import re
from nltk.corpus import stopwords

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Loading Data
def load_dataset(filepath):
    data = pd.read_csv(filepath, delimiter=";", names=["text", "emotion"])
    return data

train_data = load_dataset("train.txt")

# Text Preprocessing
def clean_text(text):
    # Lowercase the text
    text = text.lower()
    
    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    
    return text

# Apply text cleaning to the dataset
train_data['text'] = train_data['text'].apply(clean_text)

# Split data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_data["text"], train_data["emotion"], test_size=0.2, random_state=42)

# Define emotions based on your dataset
emotions = train_data["emotion"].unique()

# Tokenization and Padding
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
max_len = max(len(seq) for seq in X_train_seq)

X_train_padded = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

# Encode the target labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Modeling (Bidirectional LSTM)
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=32, input_length=max_len))
model.add(Bidirectional(LSTM(64, return_sequences=True))) 
model.add(Bidirectional(LSTM(64)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(emotions), activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Fit the model
model.fit(X_train_padded, y_train_encoded, batch_size=150, epochs=30, validation_split=0.2, callbacks=[early_stopping])

# Loop for user input
while True:
    user_input = input("Enter a text (or type 'exit' to stop): ")
    if user_input.lower() == 'exit':
        break
    
    user_input = clean_text(user_input)
    user_input_seq = tokenizer.texts_to_sequences([user_input])
    user_input_padded = pad_sequences(user_input_seq, maxlen=max_len, padding='post')

    predicted_emotion_encoded = model.predict(user_input_padded)
    predicted_emotion = le.inverse_transform([np.argmax(predicted_emotion_encoded)])
    print(f"Predicted Emotion: {predicted_emotion[0]}")
