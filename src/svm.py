import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import csv
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import re
import joblib
import os

model_filename = './model/svm/svm_model.joblib'
vectorizer_filename = './model/svm/svm_vectorizer.joblib'

df = pd.read_csv('data/washed_data.csv', header=None, encoding='utf-8')
print("load train data")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
# 20000
# Accuracy: 0.728
# 500000
# Accuracy: 0.77712

limit = 20000

X = df[5].copy()
y = df[0].copy()
# X = df.iloc[:limit, 5].copy()
# y = df.iloc[:limit, 0].copy()
# print(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
# 这个方法用于学习词汇表（vocabulary）并将文本数据转换为文档-词频矩阵（Document-Term Matrix，DTM）。
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

svm_model = LinearSVC()
svm_model.fit(X_train_vec, y_train)
y_pred = svm_model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print("Val Accuracy:", accuracy)

if not os.path.exists('./model/svm'):
    os.makedirs('./model/svm')
# 保存模型
joblib.dump(svm_model, model_filename)
joblib.dump(vectorizer, vectorizer_filename)

# 加载模型和CountVectorizer对象
loaded_model = joblib.load(model_filename)
loaded_vectorizer = joblib.load(vectorizer_filename)

# 读取测试数据
test_df = pd.read_csv('data/val_washed_data.csv',
                      header=None, encoding='utf-8')
print("load test data")
test_data = test_df[5]
test_data_vec = loaded_vectorizer.transform(test_data)
test_pred = loaded_model.predict(test_data_vec)
accuracy = accuracy_score(test_df[0], test_pred)
print("Test Accuracy:", accuracy)

# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, random_state=0)
# 注意：t-SNE 不接受稀疏矩阵，所以我们需要先转换为数组
X_2d = tsne.fit_transform(test_data_vec.toarray())

# 绘制所有样本
plt.figure(figsize=(6, 5))
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple']
for i, c in zip(range(10), colors):
    plt.scatter(X_2d[test_df[0] == i, 0],
                X_2d[test_df[0] == i, 1], c=c, label=str(i))
plt.legend()
plt.show()
