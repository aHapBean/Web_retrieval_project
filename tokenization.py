import csv
import pandas as pd
import nltk
import re

# nltk.download('punkt')

df = pd.read_csv('testdata.manual.2009.06.14.csv', header=None)
strs = df[5].tolist()
# df[0] is the label  df[5] is the data
for i in range(len(strs)):
    previous_str = strs[i]
    strs[i] = re.sub('[^\w ]', '', strs[i])     # 除去非正文内容，无意义字符
    strs[i] = strs[i].lower()
    tokenized = nltk.tokenize.word_tokenize(strs[i])

target = df[0].tolist()
print(target)
print(strs)

# 写一个SVM实现文本分类，注意对样本的分割，分割成训练集和测试集
