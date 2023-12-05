import csv
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import re
import joblib
import os


model_filename = './model/random_forest/rf_model.joblib'
vectorizer_filename = './model/random_forest/rf_vectorizer.joblib'

# file_path = 'training.1600000.processed.noemoticon.csv'
file_path = 'washed_data.csv'
df = pd.read_csv(file_path, header=None, encoding='ISO-8859-1')
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
limit = 1600000
X = df.iloc[:, 5].copy()  # 避免原地操作，使用.copy()创建新的DataFrame
y = df.iloc[:, 0].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using CountVectorizer
print("Vectorize...")
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)     # 这个方法用于学习词汇表（vocabulary）并将文本数据转换为文档-词频矩阵（Document-Term Matrix，DTM）。
X_test_vec = vectorizer.transform(X_test)

# Train the LinearSVC model
print("Train...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_vec, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test_vec)

# Print the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

if not os.path.exists('./model/random_forest'):
    os.makedirs('./model/random_forest')
# 保存模型
joblib.dump(rf_model, model_filename)
joblib.dump(vectorizer, vectorizer_filename)

# 加载模型和CountVectorizer对象
loaded_model = joblib.load(model_filename)
loaded_vectorizer = joblib.load(vectorizer_filename)

# 读取测试数据
test_df = pd.read_csv('val_washed_data.csv', header=None, encoding='ISO-8859-1')

test_data = test_df[5]  # 选择处理后的文本列
test_data_vec = loaded_vectorizer.transform(test_data)
test_pred = loaded_model.predict(test_data_vec)
accuracy = accuracy_score(test_df[0], test_pred)  # 使用原始标签列
# for i in range(len(test_df[0].tolist())):
#     if test_df[0][i] != test_pred[i]:
#         print("false class: ", test_pred[i], "err sentence: ", test_df[5][i])
# 0是悲伤
print(accuracy)

# 10000 -> acc: 73.8
# n_estimator = 100 20000 -> acc: 74.9 ( 73.53 )