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

# file_path = 'training.1600000.processed.noemoticon.csv'
file_path = 'washed_data.csv'
df = pd.read_csv(file_path, header=None, encoding='ISO-8859-1')
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
# 20000
# Accuracy: 0.728
# 500000
# Accuracy: 0.77712

limit = 100000
# combined_data = df.iloc[:limit, 5].copy()
# combined_data = df[5].copy()
# combined_data = combined_data.apply(lambda x: re.sub('[^\w ]', '', str(x).lower()))
# combined_data = combined_data.apply(lambda x: nltk.tokenize.word_tokenize(str(x)))
# combined_data = combined_data.apply(lambda x: ' '.join(x))    # NO NEED to wash again

X = df[5].copy()  # 避免原地操作，使用.copy()创建新的DataFrame
y = df.iloc[:limit, 0].copy()
print(X)

# 存储处理后的数据
# df.to_csv('washed_data.csv', index=False)
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

if not os.path.exists('./model/svm'):
    os.makedirs('./model/svm')
# 保存模型
# joblib.dump(svm_model, model_filename)
# joblib.dump(vectorizer, vectorizer_filename)


# 加载模型和CountVectorizer对象
loaded_model = joblib.load(model_filename)
loaded_vectorizer = joblib.load(vectorizer_filename)

# 读取测试数据
test_df = pd.read_csv('testdata.manual.2009.06.14.csv', header=None, encoding='ISO-8859-1')
test_df[5] = test_df[5].apply(lambda x: re.sub('[^\w ]', '', str(x).lower()))
test_df[5] = test_df[5].apply(lambda x: nltk.tokenize.word_tokenize(str(x)))
test_df[5] = test_df[5].apply(lambda x: ' '.join(x))

test_data = test_df[5]  # 选择处理后的文本列
test_data_vec = loaded_vectorizer.transform(test_data)
test_pred = loaded_model.predict(test_data_vec)
accuracy = accuracy_score(test_df[0], test_pred)  # 使用原始标签列
for i in range(len(test_df[0].tolist())):
    if test_df[0][i] != test_pred[i]:
        print("false class: ", test_pred[i], "err sentence: ", test_df[5][i])
# 0是悲伤
print(accuracy)

# 用100000个训练 acc: 75.76601671309192
# C:\Users\anaconda3\lib\site-packages\sklearn\svm\_base.py:1206: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
#   warnings.warn(
# 160万acc: 78.55

"""
在Scikit-learn库中，LinearSVC 和 SVC 是两个不同的支持向量机分类器。

LinearSVC:

LinearSVC 是支持向量机（SVM）的线性核版本。它使用线性核函数，适用于线性可分的情况，也就是数据在特征空间中可以通过一条直线进行划分。
由于使用了线性核，LinearSVC 在处理大规模数据集时通常速度较快。它对于文本分类等任务是一个常见的选择。
SVC:

SVC 则是通用的支持向量机分类器，它可以使用不同的核函数，如线性核、多项式核、径向基函数（RBF）核等。这使得它更灵活，可以处理非线性可分的情况。
由于支持不同的核函数，SVC 在处理非线性问题时可能需要更多的计算资源和时间。在处理文本数据时，通常首选使用线性核（如LinearSVC），因为文本数据往往是高维且稀疏的。

"""