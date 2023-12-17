# 计算GPT的 acc prec recall f1score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

GPT35 = '../result/GPT_3.5/output.csv'
GPT4 = '../result/GPT_4/output.csv'
GPTcur = GPT35
df = pd.read_csv(GPTcur, encoding='UTF-8')

pred = df['Prediction']
label = df['Label']
acc = accuracy_score(label, pred)
prec = precision_score(label, pred)
recall = recall_score(label, pred)
f1 = f1_score(label, pred)
print("{} result : \n accuracy: {} precision: {} recall: {} f1 score: {}".format(GPTcur, acc, prec, recall, f1))