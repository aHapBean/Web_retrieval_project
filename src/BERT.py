import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
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

limit = 160000
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
tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
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
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
val_dataset = CustomDataset(X_val_tokens, y_val)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)   # TODO the batch size ????

# 加载BERT模型
model = BertForSequenceClassification.from_pretrained('./bert-base-uncased', num_labels=2)
# model = BertForSequenceClassification.from_pretrained('./model/pretrained_model', num_labels=2)
# /bert-base-uncased
optimizer = AdamW(model.parameters(), lr=5e-5)                      # lr TODO 5e-5 -> 1e-5

device = torch.device('cuda:2')
model.to(device)

best_acc = 0.85
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
    if accuracy > best_acc:
        best_acc = accuracy
        model.save_pretrained('./model/pretrained_model')
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

# 微信有图片记录了结果
# acc 约等于 84.5
"""
the code is using the same pre-trained BERT model: 'bert-base-uncased'.

The first use is for creating a BERT tokenizer:

python
Copy code
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
This initializes a tokenizer that is designed for the 'bert-base-uncased' pre-trained model. The tokenizer is then used to tokenize the input text and convert it into a format that the BERT model can accept.

The second use is for loading the BERT model for sequence classification:

python
Copy code
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
Here, the code loads the 'bert-base-uncased' pre-trained model for sequence classification. This model is fine-tuned for a binary classification task with two labels.

So, both the tokenizer and the model are using the 'bert-base-uncased' pre-trained weights.

网络主体结构应该是相同的

BertForSequenceClassification模型是在BERT的基础上进行微调以适应文本序列分类任务的模型。微调主要涉及到调整模型的最后一层，即分类任务所需的输出层。
在BERT模型中，最后一层是一个全连接层（linear layer），该层的输出维度与分类任务中的类别数相匹配。在微调时，可以将这个全连接层调整为适应特定分类任务的输出要求。

"""