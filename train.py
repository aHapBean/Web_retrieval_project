import subprocess
import sys

# usage: python train.py svm
PREFIX = 'python'
MODEL = sys.argv[1]

if MODEL == 'svm':
    data_size_range = [10000, 20000, 50000, 100000, 200000]
    for DS in data_size_range:
        subprocess.run([PREFIX, 'svm.py', '--data_size', str(DS)])
elif MODEL == 'random_forest':
    data_size_range = [10000, 20000, 50000, 100000]
    n_estimators_range = [100, 200, 500, 1000]
    for NE in n_estimators_range:
        for DS in data_size_range:
            subprocess.run([PREFIX, 'random_forest.py',
                           '--data_size', str(DS), '--n_estimators', str(NE)])
elif MODEL == 'naive_bayes':
    data_size_range = [10000, 20000, 50000, 100000]
    for DS in data_size_range:
        subprocess.run([PREFIX, 'naive_bayes.py', '--data_size', str(DS)])
elif MODEL == 'knn':
    data_size_range = [10000, 20000, 50000, 100000]
    for DS in data_size_range:
        subprocess.run([PREFIX, 'knn.py', '--data_size', str(DS)])
else:
    print("unexpected model name")
