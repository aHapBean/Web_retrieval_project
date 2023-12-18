import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from BERT_class import CustomBertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 定义自定义数据集
class CustomDataset(Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.tokens['input_ids'][idx],
            'attention_mask': self.tokens['attention_mask'][idx],
            'labels': torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        }
    
def prepare_data():
    # nltk.download('punkt')
    # 设置文件路径和读取数据
    file_path = '../data/washed_train_data_z.csv'
    df = pd.read_csv(file_path, header=None, encoding='ISO-8859-1')
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    limit = 160000
    X = df.iloc[:limit, 5].copy()
    y = df.iloc[:limit, 0].copy()

    # val
    df_test = pd.read_csv('../data/washed_test_data_z.csv', header=None, encoding='ISO-8859-1')
    X_test = df_test.iloc[:, 5].copy()
    y_test = df_test.iloc[:, 0].copy()

    # 划分训练集和测试集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)  
    return X_train, X_val, X_test, y_train, y_val, y_test

def tokenize(X_train, X_val, X_test):
    print("start training tokenizer...")
    # 使用BERT的tokenizer将文本转换为模型可接受的格式
    tokenizer = BertTokenizer.from_pretrained('../bert-base-uncased')
    X_train_tokens = tokenizer(list(X_train), padding=True, truncation=True, return_tensors='pt', max_length=512)
    X_val_tokens = tokenizer(list(X_val), padding=True, truncation=True, return_tensors='pt', max_length=512)
    X_test_tokens = tokenizer(list(X_test), padding=True, truncation=True, return_tensors='pt', max_length=512)
    # print("X_test_tokens: ", X_test_tokens)
    # 在BERT中，BertTokenizer的作用是将输入的文本分割成词汇单元（tokens），并为每个词汇单元分配一个唯一的ID。此外，tokenizer 会添加一些特殊的标记，
    # 如[CLS]（用于表示序列的开始）和[SEP]（用于表示序列的结束），以及控制序列长度的padding标记
    return X_train_tokens, X_val_tokens, X_test_tokens
    
def get_features(model, loader):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for batch in loader:
            inputs = {key: val.to(device) for key, val in batch.items()}
            label = inputs.pop('labels').to(device)
            outputs = model(**inputs)
            features.extend(outputs.cpu().numpy())
            labels.extend(label.cpu().numpy())

    return features, labels

def plot_tsne(features, labels):
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(np.array(features))

    plt.figure(figsize=(6,6))
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    # plt.show()
    plt.savefig('256best.png')


if __name__ == '__main__':
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()

    X_train_tokens, X_val_tokens, X_test_tokens = tokenize(X_train, X_val, X_test)
    print("start trianing classifier...")

    # 创建数据加载器
    train_dataset = CustomDataset(X_train_tokens, y_train)
    val_dataset = CustomDataset(X_val_tokens, y_val)
    test_dataset = CustomDataset(X_test_tokens, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)   # TODO the batch size ????

    print("start loading model...")
    # 加载BERT模型
    model = CustomBertForSequenceClassification('../bert-base-uncased', 2)
    model.load_state_dict(torch.load('../model/bert/256-best_model.pth', map_location=torch.device('cpu')))
    optimizer = AdamW(model.parameters(), lr=5e-5)                      # lr TODO 5e-5 -> 1e-5

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print("start training...")
    # 获取训练集的特征和标签
    train_features, train_labels = get_features(model, test_loader)

    # 使用t-SNE进行可视化
    plot_tsne(train_features, train_labels)