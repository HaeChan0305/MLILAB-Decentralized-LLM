import os
import json
import jsonlines
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.io import arff
from sklearn.utils import shuffle
from datasets import load_dataset

def precessing_sst2():
    sst2 = load_dataset("stanfordnlp/sst2")
    
    train = pd.DataFrame(sst2["train"])
    test = pd.DataFrame(sst2["validation"])
    
    train = train.drop(columns=['idx'])
    test = test.drop(columns=['idx'])
    
    def match(x):
        if x == 0: return "Negative"
        elif x == 1: return "Positive"
        else: assert 0
    
    train['label'] = [match(l) for l in train['label']]
    test['label'] = [match(l) for l in test['label']]
    
    train.to_csv("./data/sst2_train.csv", index=False)
    test.to_csv("./data/sst2_test.csv", index=False)


def precessing_agnews():
    agnews = load_dataset("fancyzhx/ag_news")
    
    train = pd.DataFrame(agnews["train"])
    test = pd.DataFrame(agnews["test"])
    
    train.rename(columns = {'text' : 'sentence'}, inplace = True)
    test.rename(columns = {'text' : 'sentence'}, inplace = True)
    
    def match(x):
        if x == 0: return "World"
        elif x == 1: return "Sports"
        elif x == 2: return "Business"
        elif x == 3: return "Science/Technology"
        else: assert 0
    
    train['label'] = [match(l) for l in train['label']]
    test['label'] = [match(l) for l in test['label']]
    
    train.to_csv("./data/agnews_train.csv", index=False)
    test.to_csv("./data/agnews_test.csv", index=False)

def precessing_mr():
    mr = load_dataset("mattymchen/mr")
    mr = pd.DataFrame(mr["test"])
    
    train, test = train_test_split(mr, test_size=0.15, random_state=42)
    
    train.rename(columns = {'text' : 'sentence'}, inplace = True)
    test.rename(columns = {'text' : 'sentence'}, inplace = True)
    
    def match(x):
        if x == 0: return "Negative"
        elif x == 1: return "Positive"
        else: assert 0
    
    train['label'] = [match(l) for l in train['label']]
    test['label'] = [match(l) for l in test['label']]
    
    train.to_csv("./data/mr_train.csv", index=False)
    test.to_csv("./data/mr_test.csv", index=False)

def procesing_r8():
    train = pd.read_csv("./r8-train-stemmed.csv")
    dev = pd.read_csv("./r8-dev-stemmed.csv")
    test = pd.read_csv("./r8-test-stemmed.csv")
    train = pd.concat([train, dev])

    train.rename(columns = {'text' : 'sentence', 'intent' : 'label'}, inplace = True)
    test.rename(columns = {'text' : 'sentence', 'intent' : 'label'}, inplace = True)

    train = train.drop(columns=['edge'])
    test = test.drop(columns=['edge'])

    def match(x):
        if x == 'grain': return "Grain"
        elif x == 'earn': return "Earnings and Earnings Forecasts"
        elif x == 'interest': return "Interest Rates"
        elif x == 'money-fx': return "Money/Foreign Exchange"
        elif x == 'acq': return "Acquisitions"
        elif x == 'crude': return "Crude Oil"
        elif x == 'ship': return "Shipping"
        elif x == 'trade': return "Trade"
        else: assert 0

    train['label'] = [match(l) for l in train['label']]
    test['label'] = [match(l) for l in test['label']]

    train.to_csv("./data/r8_train.csv")
    test.to_csv("./data/r8_test.csv")

def check_class_imbalance(dataset_name):
    train = pd.read_csv(f"./data/{dataset_name}_train.csv")
    test = pd.read_csv(f"./data/{dataset_name}_test.csv")
    
    labels = set(train['label'])
    assert labels == set(test['label'])
    
    result = {label : 0 for label in labels}
    for _, row in train.iterrows():
        result[row['label']] += 1
        
    print(f"### train : {len(train)} ###")
    for k, v in result.items():
        print(f"{k} : {v} ({round(v/len(train) * 100, 1) }%)")
    print()
    print()
    
    result = {label : 0 for label in labels}
    for _, row in test.iterrows():
        result[row['label']] += 1
        
    print(f"### test : {len(test)} ###")
    for k, v in result.items():
        print(f"{k} : {v} ({round(v/len(test) * 100, 1) }%)")
    print()
    print()
        
def csv_2_jsonl(csv_path, jsonl_path):
    dataset = pd.read_csv(csv_path)
    
    processed_dataset = []
    for _, row in dataset.iterrows():
        processed_dataset.append(
            {
                "type": "chatml", 
                "messages": 
                    [
                        {"role": "system", "content": "You are a helpful assistant."}, 
                        {"role": "user", "content": row['sentence']}, 
                        {"role": "assistant", "content": row['label']}
                    ], 
                "source": "unknown"
            }
        )
        
    with open(jsonl_path, "w") as file: 
        for i in processed_dataset: file.write(json.dumps(i) + "\n")


if __name__=='__main__':
    dataset_names = ['agnews', 'mr', 'r8', 'sst2']
    for dataset_name in dataset_names:
        csv_2_jsonl(f"./data/{dataset_name}_train.csv", f"./data/{dataset_name}_train.jsonl")
        csv_2_jsonl(f"./data/{dataset_name}_test.csv", f"./data/{dataset_name}_test.jsonl") 
    