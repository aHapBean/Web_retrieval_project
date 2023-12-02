import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, TabularDataset, BucketIterator
from sklearn.model_selection import train_test_split
import spacy
import re
import nltk
import pandas as pd

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

# Create TorchText fields
TEXT = Field(tokenize='spacy', include_lengths=True)
LABEL = Field(sequential=False, use_vocab=False, dtype=torch.float)

# Create TorchText datasets
fields = [('text', TEXT), ('label', LABEL)]
train_data = TabularDataset(path='train_data.csv', format='csv', fields=fields, skip_header=False)
test_data = TabularDataset(path='test_data.csv', format='csv', fields=fields, skip_header=False)

# Build vocabulary
TEXT.build_vocab(train_data, max_size=10000, vectors='glove.6B.50d')    # Here, you are using the build_vocab method from the Field object (TEXT) to build the vocabulary for your text data. The vectors='glove.6B.50d' argument indicates that you want to use pre-trained GloVe word embeddings with 50 dimensions.
# This line downloads the pre-trained GloVe embeddings for the words in your dataset (up to the specified max_size) and associates them with the corresponding words in your vocabulary.

# Create iterators
train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=32,
    sort_key=lambda x: len(x.text),
    sort_within_batch=True
)

# Define the RNN model
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=1)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        hidden = self.fc(hidden[-1, :, :])
        return self.sigmoid(hidden)

# Instantiate the model
input_dim = len(TEXT.vocab)
embedding_dim = 50
hidden_dim = 100
output_dim = 1  # Adjust based on your data
model = RNN(input_dim, embedding_dim, hidden_dim, output_dim)

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Train the model
for epoch in range(5):  # Adjust the number of epochs based on your data
    for batch in train_iterator:
        text, text_lengths = batch.text
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, batch.label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Evaluate the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for batch in test_iterator:
        text, text_lengths = batch.text
        predictions = model(text, text_lengths).squeeze(1)
        predicted_labels = (predictions >= 0.5).float()
        total += batch.label.size(0)
        correct += (predicted_labels == batch.label).sum().item()

accuracy = correct / total
print(f'Test Accuracy: {accuracy}')
