import csv
import pandas as pd
import numpy as np
import nltk
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load and preprocess data
df = pd.read_csv('testdata.manual.2009.06.14.csv', header=None)
strs = df[5].tolist()

for i in range(len(strs)):
    previous_str = strs[i]
    strs[i] = re.sub('[^\w ]', '', strs[i])  # Remove non-alphanumeric characters
    strs[i] = strs[i].lower()
    tokenized = nltk.tokenize.word_tokenize(strs[i])
    strs[i] = ' '.join(tokenized)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(strs, df[0], test_size=0.2, random_state=42)

# Tokenize and pad sequences
max_words = 10000  # Adjust based on your data
max_len = 100  # Adjust based on your data

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

# Build the RNN model
embedding_dim = 50  # Adjust based on your data
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len))
model.add(LSTM(units=100))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f'Test Accuracy: {accuracy}')
