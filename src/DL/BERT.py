import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
import torch
import re
import nltk
from tqdm import tqdm


def main(args):
    file_path = '../data/washed_train_data.csv' 
    df = pd.read_csv(file_path, encoding='ISO-8859-1', header=None)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    limit = 5000
    X = df.iloc[:limit, 5].copy()
    y = df.iloc[:limit, 0].copy()
    y.replace(4, 1, inplace=True)

    # val  
    df_val = pd.read_csv('../data/washed_test_data.csv', encoding='ISO-8859-1', header=None)
    X_val = df_val.iloc[:, 5].copy()
    y_val = df_val.iloc[:, 0].copy()
    y_val.replace(4, 1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   

    print("start training tokenizer...")
    # 使用BERT的tokenizer将文本转换为模型可接受的格式
    tokenizer = BertTokenizer.from_pretrained('../../bert-base-uncased')
    X_train_tokens = tokenizer(list(X_train), padding=True, truncation=True, return_tensors='pt', max_length=512)
    X_test_tokens = tokenizer(list(X_test), padding=True, truncation=True, return_tensors='pt', max_length=512)

    X_val_tokens = tokenizer(list(X_val), padding=True, truncation=True, return_tensors='pt', max_length=512)
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
    model = BertForSequenceClassification.from_pretrained('../../bert-base-uncased', num_labels=2)
    optimizer = AdamW(model.parameters(), lr=args.lr)                 
    print("lr=", args.lr)

    if args.gpu != None:
        device = torch.device('cuda:' + str(args.gpu))
    else:
        device = torch.device('cpu')
    
    model.to(device)

    best_acc = 0
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as t_bar:
            for batch_idx, batch in enumerate(t_bar):
                inputs = {key: val.to(device) for key, val in batch.items()}
                outputs = model(**inputs)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()

                t_bar.set_postfix({'Training Loss': total_loss / (batch_idx + 1)})
        # 测试模型
        model.eval()
        all_preds = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Testing', unit='batch'):
                inputs = {key: val.to(device) for key, val in batch.items()}
                outputs = model(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)

        # 评估模型性能
        accuracy = accuracy_score(y_test, all_preds)
        precision = precision_score(y_test, all_preds, average='weighted')
        recall = recall_score(y_test, all_preds, average='weighted')
        f1 = f1_score(y_test, all_preds, average='weighted')

        print(f'Validating Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
        if accuracy > best_acc:
            best_acc = accuracy
            # model.save_pretrained('./model/pretrained_model')
            # Load the trained model
            # loaded_model = BertForSequenceClassification.from_pretrained('path_to_save_model')
            # Move the loaded model to the device (GPU or CPU) you want to use
            # loaded_model.to(device)
        
        all_preds = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Valdating', unit='batch'):
                inputs = {key: val.to(device) for key, val in batch.items()}
                outputs = model(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)

        accuracy = accuracy_score(y_val, all_preds)
        precision = precision_score(y_val, all_preds, average='weighted')
        recall = recall_score(y_val, all_preds, average='weighted')
        f1 = f1_score(y_val, all_preds, average='weighted')

        print(f'Testing Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

import argparse
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--gpu', type=int, default=None)
    args = parser.parse_args()
    return args
    

if __name__ == '__main__':
    args = args_parser()
    main(args)