import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AlbertTokenizer, AlbertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
import torch
import re
import nltk
from tqdm import tqdm

# nltk.download('punkt')
# 设置文件路径和读取数据
file_path = 'washed_data.csv'
df = pd.read_csv(file_path, header=None, encoding='ISO-8859-1')
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

limit = 10000
X = df.iloc[:limit, 5].copy()
y = df.iloc[:limit, 0].copy()

# val
df_val = pd.read_csv('val_washed_data.csv', header=None, encoding='ISO-8859-1')
X_val = df_val.iloc[:, 5].copy()
y_val = df_val.iloc[:, 0].copy()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   

print("start training tokenizer...")
# 使用BERT的tokenizer将文本转换为模型可接受的格式
tokenizer = AlbertTokenizer.from_pretrained('./albert-large-v2', local_files_only=True)
X_train_tokens = tokenizer(list(X_train), padding=True, truncation=True, return_tensors='pt', max_length=512)
X_test_tokens = tokenizer(list(X_test), padding=True, truncation=True, return_tensors='pt', max_length=512)

X_val_tokens = tokenizer(list(X_val), padding=True, truncation=True, return_tensors='pt', max_length=512)
# print("X_test_tokens: ", X_test_tokens)
# 在BERT中，BertTokenizer的作用是将输入的文本分割成词汇单元（tokens），并为每个词汇单元分配一个唯一的ID。此外，tokenizer 会添加一些特殊的标记，
# 如[CLS]（用于表示序列的开始）和[SEP]（用于表示序列的结束），以及控制序列长度的padding标记
print("start trianing classifier...")

# print("tokenizer: ", X_train_tokens)

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
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
val_dataset = CustomDataset(X_val_tokens, y_val)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)   # TODO the batch size ????

# 加载BERT模型
pretrain = None
origin = './albert-large-v2'
model = AlbertForSequenceClassification.from_pretrained(origin, num_labels=2)
optimizer = AdamW(model.parameters(), lr=5e-5)                      # lr TODO 5e-5 -> 1e-5

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
model.to(device)

# best_acc = 0.83
# 0.8346
epochs = 20
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
    print(f'Testing Accuracy: {accuracy:.4f}')
    # if accuracy > best_acc:
    #     best_acc = accuracy
    #     model.save_pretrained('./model/pretrained_model')
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

    # 评估模型性能
    accuracy = accuracy_score(y_val, all_preds)
    print(f'Validating Accuracy: {accuracy:.4f}')
    
# 准确率没有提升0.5 找找问题
# 