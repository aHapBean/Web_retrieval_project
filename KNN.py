import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import re
import joblib
import os

model_filename = './model/knn/knn_model.joblib'
vectorizer_filename = './model/knn/knn_vectorizer.joblib'

file_path = 'washed_data.csv'
df = pd.read_csv(file_path, header=None, encoding='ISO-8859-1')
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
limit = 100000
X = df.iloc[:limit, 5].copy()
y = df.iloc[:limit, 0].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using CountVectorizer
print("Vectorize...")
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the KNN model
print("Train...")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_vec, y_train)

# Make predictions on the test set
y_pred = knn_model.predict(X_test_vec)

# Print the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

if not os.path.exists('./model/knn'):
    os.makedirs('./model/knn')
# Save the model
joblib.dump(knn_model, model_filename)
joblib.dump(vectorizer, vectorizer_filename)

# Load the model and CountVectorizer object
loaded_model = joblib.load(model_filename)
loaded_vectorizer = joblib.load(vectorizer_filename)

# Read test data
test_df = pd.read_csv('val_washed_data.csv', header=None, encoding='ISO-8859-1')

test_data = test_df[5]
test_data_vec = loaded_vectorizer.transform(test_data)
test_pred = loaded_model.predict(test_data_vec)
accuracy = accuracy_score(test_df[0], test_pred)
print(accuracy)
# 100000 n_es = 5 : acc 64
