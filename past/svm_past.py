# -*- coding: utf-8 -*-
import csv
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import re

# nltk.download('punkt')

# Read CSV file into a DataFrame

file_path = 'training.1600000.processed.noemoticon.csv'
df = pd.read_csv(file_path, header=None, encoding='ISO-8859-1')
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
# frac=1表示保持所有行，random_state是一个种子值，用于确保每次运行时都会得到相同的随机顺序，reset_index(drop=True)用于重置索引并丢弃旧的索引。
# 20000
# Accuracy: 0.728

# 500000
# C:\Users\anaconda3\lib\site-packages\sklearn\svm\_base.py:1206: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
#   warnings.warn(
# Accuracy: 0.77712

# df = pd.read_csv('testdata.manual.2009.06.14.csv', header=None, encoding='ascii')

limit = 500000


# first_part = df.iloc[:limit, 5].copy()
# last_part = df.iloc[-limit:, 5].copy()
# combined_data = pd.concat([first_part, last_part])
combined_data = df.iloc[:limit, 5].copy()

# 仅对部分行进行文本预处理
combined_data = combined_data.apply(lambda x: re.sub('[^\w ]', '', str(x).lower()))
combined_data = combined_data.apply(lambda x: nltk.tokenize.word_tokenize(str(x)))
combined_data = combined_data.apply(lambda x: ' '.join(x))

# 将数据分为特征（X）和目标标签（y）
X = combined_data.copy()  # 避免原地操作，使用.copy()创建新的DataFrame
# y = pd.concat([df.iloc[:limit, 0], df.iloc[-limit:, 0]])
y = df.iloc[:limit, 0].copy()
print(X)

# 存储处理后的数据
# df.to_csv('washed_data.csv', index=False)

print(len(X))
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)     # 这个方法用于学习词汇表（vocabulary）并将文本数据转换为文档-词频矩阵（Document-Term Matrix，DTM）。
X_test_vec = vectorizer.transform(X_test)

# Train the LinearSVC model
svm_model = LinearSVC()
svm_model.fit(X_train_vec, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test_vec)

# Print the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
