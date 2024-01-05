# CS3307 Internet Information Extraction

## Introduction 介绍
情感分析是一种文本分析方法，利用自然语言处理技术从文本中提取信息和语义，确定作者的态度。这包括抽取实体、属性、观点、观点持有者、时间等要素，以分析文本对商品评价、明星舆论等的情感。

本项目使用推特评论数据（160万条训练集，360条测试集），采用K近邻、支持向量机、朴素贝叶斯和随机森林等传统机器学习模型，以不同方式识别文本情感倾向。同时，尝试了使用Transformer的Bert模型，并对其进行了改进以在理解语境和语义信息方面表现出色。最后，项目还探索了大语言模型ChatGPT，并利用prompt engineering技术引导模型生成更准确的情感分析结果，同时研究了不同数据清洗方法对分类效果的影响。


## Requirement 项目环境

```bash
pip install -r requirement.txt
```

## Run 项目运行

### 准备

#### 模型准备
在huggingface下载`bert-base-uncased`文件，包括
- config.json
- pytorch_model.bin
- tokenizer_config.json
- tokenizer.jon
- vocab.txt

放在`projectxx/`目录下，即与`projectxx/src`同级目录。

#### 数据准备
将训练数据改名为train.csv，测试数据改为test.csv，放在`./src/data目录中`，更改`./src/data/wash_data_base.py`参数后运行`wash_data_base.py`即可得到清洗后的数据`washed_train_data.csv`与`washed_test_data.csv`。

### 传统机器学习方法

进入`./src/traditional_ML`，使用`KNN, Naive bayes, random forest, svm`四种机器学习方法，运行指令：

```bash
python KNN.py 
python Naive_bayes.py
python random_forest.py
python svm.py
```

### 深度学习方法
进入`./src/DL`，使用预训练语言模型Bert，添加两层线性层进行微调训练，运行指令

#### 模型训练

```bash
python BERT.py --gpu [gpu_id]
```

#### 模型测试

如果你想直接测试本项目训练出的最高性能模型，请直接运行
```bash
python BERT.py --gpu [gpu_id] --test-only
```
运行这一步时，请保证`32-best_model.pth`文件在同级目录下。

最高性能： `Testing Accuracy: 0.8802, Precision: 0.8823, Recall: 0.8802, F1: 0.8801`

### 大语言模型方法
进入`./src/LLM`，使用`GPT-3.5/4`进行文本分类。

自行配置`OPEN_API_KEY`等信息后即可运行文件`openai-test.py`

`GPT-4`文本分类性能：`Testing Accuracy: 0.9387186629526463 Precision: 0.9081632653061225 Recall: 0.978021978021978 F1 score: 0.9417989417989417`

### project structures TODO update
```
.
├── README.md
├── __pycache__
│   └── BERT_class.cpython-311.pyc
├── bert-base-uncased
│   ├── bert-base-uncased.py
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── vocab.txt
├── data
│   ├── washed_test_data_s.csv
│   ├── washed_test_data_z.csv
│   ├── washed_train_data_s.csv
│   └── washed_train_data_z.csv
├── doc
│   └── project.pdf
├── model
│   ├── bert
│   │   └── best_model.pth
│   ├── knn
│   │   ├── knn_model.joblib
│   │   └── knn_vectorizer.joblib
│   ├── naive_bayes
│   │   ├── bayes_vectorizer.joblib
│   │   └── nb_moble.joblib
│   └── svm
│       ├── svm_model.joblib
│       └── svm_vectorizer.joblib
├── past
│   ├── encoding_detect.py
│   └── svm_past.py
├── raw_data
│   ├── testdata.manual.2009.06.14.csv
│   └── training.1600000.processed.noemoticon.csv
├── result
│   ├── bert
│   │   ├── best_acc.txt
│   │   ├── lr=1e-5.txt
│   │   ├── lr=2e-5.txt
│   │   ├── lr=3e-5.txt
│   │   ├── lr=4e-5.txt
│   │   └── lr=5e-5.txt
│   ├── random_forest
│   │
│   └── svm
│
└── src
    ├── ALBERT.py
    ├── BERT+SNE.py
    ├── BERT.py
    ├── BERT_class.py
    ├── BERTmy.py
    ├── KNN.py
    ├── Naive_bayes.py
    ├── bert.ipynb
    ├── random_forest.py
    ├── svm.py
    ├── tokenization.py
    ├── train.py
    └── wash_data.ipynb
```