import argparse
from datetime import datetime
import pandas as pd
import joblib
import os
from distutils.util import strtobool
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def print_time():
    now = datetime.now()
    time_string = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Current date and time:", time_string)

def train():
    print_time()
    print("Read data...")
    file_path = '../data/washed_train_data_s.csv'
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    limit = 320000
    X = df.iloc[:limit, 1].copy()
    y = df.iloc[:limit, 0].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize the text data using CountVectorizer
    print_time()
    print("Vectorize...")
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train the KNN model
    print_time()
    print("Train...")
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_vec, y_train)

    print_time()
    print("Predict...")
    # Make predictions on the test set
    y_pred = knn_model.predict(X_test_vec)

    print_time()
    print("finish prediction")
    # Print the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print("validation set:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    return knn_model, vectorizer

def load_model():
    # Load the model and CountVectorizer object
    model_filename = '../model/knn/knn_model.joblib'
    vectorizer_filename = '../model/knn/knn_vectorizer.joblib'
    knn_model = joblib.load(model_filename)
    vectorizer = joblib.load(vectorizer_filename)
    return knn_model, vectorizer


def predict(knn_model, vectorizer):
    # Read test data
    test_df = pd.read_csv('../data/washed_test_data_s.csv', encoding='ISO-8859-1')

    # test_ground_truth = test_df.iloc[:, 0].copy()
    test_data = test_df.iloc[:, 1].copy()
    test_data_vec = vectorizer.transform(test_data)
    test_pred = knn_model.predict(test_data_vec)
    print("test set:")
    accuracy = accuracy_score(test_df['labels'], test_pred)
    precision = precision_score(test_df['labels'], test_pred)
    recall = recall_score(test_df['labels'], test_pred)
    f1 = f1_score(test_df['labels'], test_pred)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

def save_model(knn_model, model_filename, vectorizer, vectorizer_filename):
    # Save the model
    if not os.path.exists('../model/knn'):
        os.makedirs('../model/knn')

    joblib.dump(knn_model, model_filename)
    joblib.dump(vectorizer, vectorizer_filename)

def main(args):
    if args.train == True:
        model, vectorizer = train()
        model_filename = '../model/knn/knn_model.joblib'
        vectorizer_filename = '../model/knn/knn_vectorizer.joblib'
        save_model(model, model_filename, vectorizer, vectorizer_filename)
    else:
        model, vectorizer = load_model()
        predict(model, vectorizer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default=True, help='train(True) or predict(False), default train')
    args = parser.parse_args()
    args.train = strtobool(args.train)
    print(f"args.train is {args.train}")
    main(args)

# 100000 n_es = 5 : acc 64

# 100000 n_es 8:2
# val set acc 66.515%
# test set acc 64.903%