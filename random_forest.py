import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import datetime
import joblib
import os
import warnings

# ignore warnings
warnings.filterwarnings('ignore')

model_filename = './model/random_forest/random_forest_model.joblib'
vectorizer_filename = './model/random_forest/random_forest_vectorizer.joblib'
train_file_path = 'washed_train_data.csv'
test_file_path = 'washed_test_data.csv'


def write_result(val_acc, val_auc, val_f1, test_acc, test_auc, test_f1, args):
    file_name = 'result/random_forest/' + \
        datetime.date.today().strftime('%Y-%m-%d') + '_.log'
    if not os.path.exists('./result/random_forest/'):
        os.makedirs('./result/random_forest/')
    with open(file_name, 'a') as f:
        f.write("data_size: " + str(args.data_size) + '\n' +
                "val acc: %.2f%%" % (val_acc * 100.0) + '\t\t' +
                "val auc: %.2f%%" % (val_auc * 100.0) + '\t\t' +
                "val f1: %.2f%%" % (val_f1 * 100.0) + '\n' +
                "test acc: %.2f%%" % (test_acc * 100.0) + '\t' +
                "test auc: %.2f%%" % (test_auc * 100.0) + '\t' +
                "test f1: %.2f%%" % (test_f1 * 100.0) + '\n')


def main(args):
    train_df = pd.read_csv(train_file_path, header=None, encoding='ISO-8859-1')
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    # print("train data loaded")

    data_size = args.data_size
    X = train_df.iloc[:data_size, 5].copy()
    y = train_df.iloc[:data_size, 0].copy()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # get embeddings
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_val)

    # set and train model
    model = RandomForestClassifier(
        n_estimators=100, random_state=42)
    model.fit(X_train_vec, y_train)

    # validate
    y_val_pred = model.predict(X_test_vec)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    print("val acc: %.2f%%" % (val_acc * 100.0), '\t',
          "val auc: %.2f%%" % (val_auc * 100.0), '\t',
          "val f1: %.2f%%" % (val_f1 * 100.0))

    # save model and CountVectorizer object
    if args.save_model:
        if not os.path.exists('./model/random_forest'):
            os.makedirs('./model/random_forest')
        joblib.dump(model, model_filename)
        joblib.dump(vectorizer, vectorizer_filename)

    # test
    test_df = pd.read_csv(test_file_path, header=None, encoding='ISO-8859-1')
    # print("test data loaded")

    # # load model and CountVectorizer object
    # svm_model = joblib.load(model_filename)
    # vectorizer = joblib.load(vectorizer_filename)

    X_test = test_df[5]
    y_test = test_df[0]
    X_test_vec = vectorizer.transform(X_test)
    y_test_pred = model.predict(X_test_vec)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    print("test acc: %.2f%%" % (test_acc * 100.0), '\t',
          "test auc: %.2f%%" % (test_auc * 100.0), '\t',
          "test f1: %.2f%%" % (test_f1 * 100.0), '\n')
    # # show misclassified sentences
    # for i in range(len(y_test.tolist())):
    #     if y_test[i] != y_test_pred[i]:
    #         print("false class: ", y_test_pred[i], "err sentence: ", test_df[5][i])
    write_result(val_acc, val_auc, val_f1, test_acc, test_auc, test_f1, args)

    return test_acc, test_auc, test_f1


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_size', type=int, default=100000)
    parser.add_argument('--save_model', type=bool, default=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print("model: random_forest",
          "\tdata_size: ", args.data_size,)
    test_acc, test_auc, test_f1 = main(args)
