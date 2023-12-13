# CS3307 Internet Information Extraction

file structures
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