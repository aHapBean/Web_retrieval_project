import csv
import pandas as pd
import nltk
import re

# nltk.download('punkt')

df = pd.read_csv('training.1600000.processed.noemoticon.csv', header=None, encoding='ISO-8859-1')
# df = pd.read_csv('testdata.manual.2009.06.14.csv', header=None, encoding='ISO-8859-1')
# df.iloc[:, 5] = df.iloc[:, 5].apply(lambda x: re.sub('[^\w ]', '', str(x).lower()))
df.iloc[:, 5] = df.iloc[:, 5].apply(lambda x: nltk.tokenize.word_tokenize(str(x)))
df.iloc[:, 5] = df.iloc[:, 5].apply(lambda x: ' '.join(x))    # no need to wash
df.iloc[:, 0] = df.iloc[:, 0].replace(4, 1) # NOTE
df.to_csv('raw_washed_data.csv', index=False, header=None)
# df.to_csv('raw_val_washed_data.csv', index=False, header=None)

# wash data code NOTE the file name !!!!
# df.iloc[:, 5] = df.iloc[:, 5].apply(lambda x: re.sub('[^\w ]', '', str(x).lower()))
# df.iloc[:, 5] = df.iloc[:, 5].apply(lambda x: nltk.tokenize.word_tokenize(str(x)))
# df.iloc[:, 5] = df.iloc[:, 5].apply(lambda x: ' '.join(x))    # no need to wash
# df.iloc[:, 0] = df.iloc[:, 0].replace(4, 1) # NOTE
# df.to_csv('washed_....csv', index=False, header=None)

# 写一个SVM实现文本分类，注意对样本的分割，分割成训练集和测试集

# encoding='ISO-8859-1'