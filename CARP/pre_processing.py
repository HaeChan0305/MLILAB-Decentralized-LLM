import os
import argparse
import json
import jsonlines
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.io import arff
from random import shuffle
from datasets import load_dataset

CHOICES = {
    "agnews" : ["Sports", "World", "Science/Technology", "Business"],
    "mr" : ["Positive", "Negative"],
    "r8" : ["Grain", "Earnings and Earnings Forecasts", "Interest Rates", "Money/Foreign Exchange", "Acquisitions", "Crude Oil", "Shipping", "Trade"],
    "sst2" : ["Positive", "Negative"],
}

KEYWORDS = {
    "agnews" : "TOPIC",
    "mr" : "SENTIMENT",
    "r8" : "TOPIC",
    "sst2" : "SENTIMENT",
}

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
    
def check_class_imbalance_jsonl(jsonl_path, dataset_name):
    dataset = []
    with jsonlines.open(jsonl_path) as file:
        for line in file:
            dataset.append(line)
    
    result = {label : 0 for label in CHOICES[dataset_name]}
    for example in dataset:
        result[example['messages'][-1]['content']] += 1
        
    print(f"### train : {len(dataset)} ###")
    for k, v in result.items():
        print(f"{k} : {v} ({round(v/len(dataset) * 100, 1) }%)")
    print()
    print()
        
def make_prompt(dataset_name, sentence):
    prompt = """Classify the {0} of the input sentence as {1}.\nINPUT : {2}\nOUTPUT : """
    c = ', '.join(CHOICES[dataset_name][:-1]) + " or " + CHOICES[dataset_name][-1]
    return prompt.format(KEYWORDS[dataset_name].lower(), c, sentence)
    

def csv_2_jsonl(csv_path, jsonl_path, dataset_name):
    dataset = pd.read_csv(csv_path)
    
    processed_dataset = []
    
    for _, row in dataset.iterrows():
        processed_dataset.append(
            {
                "type": "chatml", 
                "messages": 
                    [
                        {"role": "system", "content": "You are a helpful assistant."}, 
                        {"role": "user", "content": make_prompt(dataset_name, row['sentence'])}, 
                        {"role": "assistant", "content": row['label']}
                    ], 
                "source": "unknown"
            }
        )
        
    with open(jsonl_path, "w") as file: 
        for i in processed_dataset: file.write(json.dumps(i) + "\n")


def split_train_data_iid(jsonl_path, dataset_name):
    train = []
    with jsonlines.open(jsonl_path) as file:
        for line in file:
            train.append(line)
    
    dataset_per_labels = {}
    for choice in CHOICES[dataset_name]:
        dataset_per_labels[choice] = [example for example in train if example['messages'][-1]['content'] == choice]

    dataset_1 = []
    dataset_2 = []
    for dataset in dataset_per_labels.values():
        dataset_1 += dataset[:len(dataset)//2]
        dataset_2 += dataset[len(dataset)//2:]
        
    shuffle(dataset_1)
    shuffle(dataset_2)
    
    with open(jsonl_path.replace("train.jsonl", "train_1.jsonl") , "w") as file: 
        for i in dataset_1: file.write(json.dumps(i) + "\n")
        
    with open(jsonl_path.replace("train.jsonl", "train_2.jsonl") , "w") as file: 
        for i in dataset_2: file.write(json.dumps(i) + "\n")


def preprocessing_for_debate_same_checkpoint(dataset_name, checkpoint, client1, client2, r):
    test = []
    
    if r == 1:
        prev_test_path = f"./data/{dataset_name}_test.jsonl"
        prev_result_path = f"{dataset_name}_result.json"
    else:
        prev_test_path = f"./output_5_1_{client1}/{dataset_name}/checkpoint-{checkpoint}/{dataset_name}_test_round_{r - 1}.jsonl"
        prev_result_path = f"{dataset_name}_result_round_{r - 1}.json"
        
    with jsonlines.open(prev_test_path) as file:
        for line in file:
            test.append(line)


    with open(f"./output_5_1_{client1}/{dataset_name}/checkpoint-{checkpoint}/{prev_result_path}", "r") as file:
        client_1_data = json.load(file)
        client_1_answers = [example['prediction'] for example in client_1_data] 
        
    with open(f"./output_5_1_{client2}/{dataset_name}/checkpoint-{checkpoint}/{prev_result_path}", "r") as file:
        client_2_data = json.load(file)
        client_2_answers = [example['prediction'] for example in client_2_data] 

    assert len(test) == len(client_1_answers)
    assert len(test) == len(client_2_answers)

    new_test = []
    for t, c1, c2 in zip(test, client_1_answers, client_2_answers):
        t['messages'].insert(-1, {"role" : "assistant",
                                  "content" : c1})
        
        t['messages'].insert(-1, {"role" : "user",
                                  "content" : f"This is the recent/updated answer from another agent: {c2}. Use this answer carefully as additional advice, {t['messages'][1]['content']}"})
        
        new_test.append(t)

    with open(f"./output_5_1_{client1}/{dataset_name}/checkpoint-{checkpoint}/{dataset_name}_test_round_{r}.jsonl" , encoding= "utf-8",mode="w") as file: 
        for i in new_test: file.write(json.dumps(i) + "\n")
        
def preprocessing_for_debate(dataset_name, client1, client2, r):
    test = []
    
    if r == 1:
        prev_test_path = f"./data/{dataset_name}_test.jsonl"
        prev_result_path = f"{dataset_name}_result.json"
    else:
        prev_test_path = f"./output_5_2_{client1}/{dataset_name}/{dataset_name}_test_round_{r - 1}.jsonl"
        prev_result_path = f"{dataset_name}_result_round_{r - 1}.json"
        
    with jsonlines.open(prev_test_path) as file:
        for line in file:
            test.append(line)

    with open(f"./output_5_2_{client1}/{dataset_name}/{prev_result_path}", "r") as file:
        client_1_data = json.load(file)
        client_1_answers = [example['prediction'] for example in client_1_data] 
        
    with open(f"./output_5_2_{client2}/{dataset_name}/{prev_result_path}", "r") as file:
        client_2_data = json.load(file)
        client_2_answers = [example['prediction'] for example in client_2_data] 

    assert len(test) == len(client_1_answers)
    assert len(test) == len(client_2_answers)

    new_test = []
    for t, c1, c2 in zip(test, client_1_answers, client_2_answers):
        t['messages'].insert(-1, {"role" : "assistant",
                                  "content" : c1})
        
        t['messages'].insert(-1, {"role" : "user",
                                  "content" : f"This is the recent/updated answer from another agent: {c2}. Use this answer carefully as additional advice, {t['messages'][1]['content']}"})
        
        new_test.append(t)

    with open(f"./output_2_{client1}/{dataset_name}/{dataset_name}_test_round_{r}.jsonl" , encoding= "utf-8",mode="w") as file: 
        for i in new_test: file.write(json.dumps(i) + "\n")
        


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument("-d", "--dataset-name", type=str, choices=['agnews', 'mr', 'r8', 'sst2'])
    # parser.add_argument("-c", "--checkpoint", type=int, help="Checkpoint path")
    parser.add_argument("-c1", "--client1", type=int)
    parser.add_argument("-c2", "--client2", type=int)
    parser.add_argument("-r", "--round", type=int)

    args = parser.parse_args()
    
    preprocessing_for_debate(args.dataset_name, args.client1, args.client2, args.round)