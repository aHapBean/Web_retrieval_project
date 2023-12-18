import csv
import os
import openai
import pandas as pd
import datetime
from tqdm import tqdm

os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ["HTTP_PROXY"] = 'http://127.0.0.1:7890'
api_key = os.environ.get("OPENAI_API_KEY")

client = openai.OpenAI(api_key=api_key)
df = pd.read_csv('washed_test_data.csv')
texts = df['datas'].tolist()
labels = df['labels'].tolist()

acc_cnt = 0
filename = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + '.csv'
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Prediction", "Label", "Text"])

for i in tqdm(range(len(texts))):
    completion = client.chat.completions.create(
        # model="gpt-3.5-turbo",
        model="gpt-4",
        # model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "You are a master of sentiment analysis, adept at distinguishing whether people's speech is positive or negative."},
            {"role": "user", "content": "I will give you a group of people's comments on Twitter. \
                You need to judge whether their comments are positive or negative. \
                You only need to answer 1 as positive or 0 as negative  in order."},
            # ---selective prompt---
            {"role": "user", "content": "Example: this week is not going as i had hoped"},
            {"role": "assistant", "content": "0"},
            # ----------------------
            {"role": "user", "content": texts[i]},
        ]
    )
    out = completion.choices[0].message.content
    out = out.lower()
    if '1' in out or 'positive' in out:
        result = 1
    elif '0' in out or 'negative' in out:
        result = 0
    else:
        result = -1
    if result == labels[i]:
        acc_cnt += 1
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([result, labels[i], texts[i]])

print("Accuracy: ", acc_cnt / len(texts))
