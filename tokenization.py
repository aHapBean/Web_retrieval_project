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

# wash data code NOTE the file name !!!!
# df.iloc[:, 5] = df.iloc[:, 5].apply(lambda x: re.sub('[^\w ]', '', str(x).lower()))
# df.iloc[:, 5] = df.iloc[:, 5].apply(lambda x: nltk.tokenize.word_tokenize(str(x)))
# df.iloc[:, 5] = df.iloc[:, 5].apply(lambda x: ' '.join(x))    # no need to wash
# df.iloc[:, 0] = df.iloc[:, 0].replace(4, 1) # NOTE
# df.to_csv('washed_....csv', index=False, header=None)

# 写一个SVM实现文本分类，注意对样本的分割，分割成训练集和测试集

# encoding='ISO-8859-1'