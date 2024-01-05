import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer, AdamW
from torch.utils.data import DataLoader, Dataset
from BERT_class import CustomBertForSequenceClassification 
import torch
import torch.nn as nn
import re
import nltk
from tqdm import tqdm

def main(args):
    file_path = '../data/washed_train_data.csv'
    df = pd.read_csv(file_path, header=None, encoding='ISO-8859-1')
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    limit = 5000
    if args.test_only:
        limit = 100
    X = df.iloc[:limit, 5].copy()
    y = df.iloc[:limit, 0].copy()

    # val
    df_val = pd.read_csv('../data/washed_test_data.csv', header=None, encoding='ISO-8859-1')
    X_val = df_val.iloc[:, 5].copy()
    y_val = df_val.iloc[:, 0].copy()
    if args.test_only:
        X_val = X_val.str.replace(' no ', ' ')
        X_val = X_val.str.replace('no ', ' ')
        X_val = X_val.str.replace(' no', ' ')
    
    

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   

    print("start training tokenizer...")
    # 使用BERT的tokenizer将文本转换为模型可接受的格式
    max_length = args.max_size
    tokenizer = BertTokenizer.from_pretrained('../../bert-base-uncased')
    X_train_tokens = tokenizer(list(X_train), padding=True, truncation=True, return_tensors='pt', max_length=max_length)
    X_test_tokens = tokenizer(list(X_test), padding=True, truncation=True, return_tensors='pt', max_length=max_length)

    X_val_tokens = tokenizer(list(X_val), padding=True, truncation=True, return_tensors='pt', max_length=max_length)
    print("start trianing classifier...")
    
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

    # 创建数据加载器
    train_dataset = CustomDataset(X_train_tokens, y_train)
    test_dataset = CustomDataset(X_test_tokens, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    val_dataset = CustomDataset(X_val_tokens, y_val)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)   

    # 加载BERT模型
    model = CustomBertForSequenceClassification('../../bert-base-uncased', 2)
    optimizer = AdamW(model.parameters(), lr=4e-5)                      

    if args.gpu != None:
        device = torch.device('cuda:' + args.gpu)
    else:
        device = torch.device('cpu')
    model.to(device)
    # with open("best_acc.txt", "r") as f:
    #     best_acc = f.read()
    #     best_acc = float(best_acc)
    best_acc = 0 # need to be
    # 0.8346
    epochs = 5
    if args.test_only:
        all_preds = []
        model.eval()
        # model.load_state_dict(torch.load('../model/best_model.pth'))
        model.load_state_dict(torch.load('./32-best_model.pth'))     # NOTE the best one
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Testing', unit='batch'):
                inputs = {key: val.to(device) for key, val in batch.items()}
                labels = inputs.pop('labels').to(device)  # 提取并将标签移动到设备
                logits = model(**inputs)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
        # for i in range(len(all_preds)):
        #     if all_preds[i] != y_val[i]:
        #         # print("corrent index {}, sentence: {}, \nit is: {}".format(i, X_val[i], y_val[i]))
        #         print("error index {}, sentence: {}, \nit should be: {}".format(i, X_val[i], y_val[i]))
        accuracy = accuracy_score(y_val, all_preds)
        precision = precision_score(y_val, all_preds, average='weighted')
        recall = recall_score(y_val, all_preds, average='weighted')
        f1 = f1_score(y_val, all_preds, average='weighted')

        print(f'Testing Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
        return 
    tmp_best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as t_bar:
            for batch_idx, batch in enumerate(t_bar):
                inputs = {key: val.to(device) for key, val in batch.items()}
                labels = inputs.pop('labels').to(device)  # 提取并将标签移动到设备
                outputs = model(**inputs)
                logits = outputs
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                t_bar.set_postfix({'Training Loss': total_loss / (batch_idx + 1)})

        # 评估模型性能（测试集）
        model.eval()
        all_preds = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Validating', unit='batch'):
                inputs = {key: val.to(device) for key, val in batch.items()}
                labels = inputs.pop('labels').to(device)  # 提取并将标签移动到设备
                logits = model(**inputs)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)

        accuracy = accuracy_score(y_test, all_preds)
        precision = precision_score(y_test, all_preds, average='weighted')
        recall = recall_score(y_test, all_preds, average='weighted')
        f1 = f1_score(y_test, all_preds, average='weighted')

        print(f'Valdating Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

        # 评估模型性能（验证集）
        all_preds = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating', unit='batch'):
                inputs = {key: val.to(device) for key, val in batch.items()}
                labels = inputs.pop('labels').to(device)  # 提取并将标签移动到设备
                logits = model(**inputs)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
            
        accuracy = accuracy_score(y_val, all_preds)
        precision = precision_score(y_val, all_preds, average='weighted')
        recall = recall_score(y_val, all_preds, average='weighted')
        f1 = f1_score(y_val, all_preds, average='weighted')

        print(f'Testing Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

        if accuracy > tmp_best_acc:
            tmp_best_acc = accuracy
            # torch.save(model.state_dict(), '../model/' + str(args.max_size) + '-best_model.pth')
        
        if accuracy > best_acc:
            best_acc = accuracy
            # 保存最佳模型
            print("best_acc upd: ", best_acc)
            # with open("best_acc.txt", "w") as f:
            #     f.write(str(best_acc))
            # torch.save(model.state_dict(), './model/best_model.pth')

import argparse
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=4e-5)
    parser.add_argument('--test-only', action='store_true', default=False)
    parser.add_argument('--max-size', type=int, default=32)
    parser.add_argument('--gpu', default=None, type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = args_parser()
    print(args)
    main(args)
    