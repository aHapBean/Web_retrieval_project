import subprocess
import sys

# python train.py svm
PREFIX = 'python'
MODEL = sys.argv[1]

data_size_range = [10000, 20000, 50000, 100000, 200000]
if MODEL == 'svm':
    for DS in data_size_range:
        subprocess.run([PREFIX, 'svm.py', '--data_size', str(DS)])
