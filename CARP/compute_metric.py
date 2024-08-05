import os
import re
import json
import argparse
import numpy as np

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

def check_result_length(data, dataset_name):
    truth_table = {
        "agnews" : 7600,
        "mr" : 1600,
        "r8" : 2189,
        "sst2" : 872
    }
    
    if dataset_name in truth_table.keys():
        if len(data) != truth_table[dataset_name]:
            print(f"{dataset_name} length is {len(data)}. It should be {truth_table[dataset_name]}.")
            return False
        else:
            return True
    
    else:
        print("Unknown dataset", dataset_name)
        return False

def harsh_extract_answer(dataset_name, completion):
    pattern_table = {
        "agnews" : r'\bTopic\b\s*:\s*(Sports|World|Science/Technology|Business)\b',
        "mr" : r'\bSentiment\b\s*:\s*(Positive|Negative)\b',
        "r8" : r'\bTopic\b\s*:\s*(Grain|Earnings and Earnings Forecasts|Interest Rates|Money/Foreign Exchange|Acquisitions|Crude Oil|Shipping|Trade)\b',
        "sst2" : r'\bSentiment\b\s*:\s*(Positive|Negative)\b',
    }

    pattern = pattern_table[dataset_name]
    matches = re.findall(pattern, completion, re.IGNORECASE)
    if len(matches) == 0:
        return INVALID_ANS
    else:
        return matches[-1]
    
def naive_extract_answer(dataset_name, completion):
    pattern_table = {
        "agnews" : ["Sports", "World", "Science/Technology", "Business"],
        "mr" : ["Positive", "Negative"],
        "r8" : ["Grain", "Earnings and Earnings Forecasts", "Interest Rates", "Money/Foreign Exchange", "Acquisitions", "Crude Oil", "Shipping", "Trade"],
        "sst2" : ["Positive", "Negative"],
    }

    pattern = pattern_table[dataset_name]
    matches = []
    for word in pattern:
        if word.lower() in completion.lower():
            matches.append(word)

    if len(matches) == 1:
        return matches[-1]
    else:
        return INVALID_ANS + str(len(matches))


def extract_answer(dataset_name, completion):
    answer = harsh_extract_answer(dataset_name, completion)
    if answer != INVALID_ANS:
        return answer
    return naive_extract_answer(dataset_name, completion)

def is_correct(t, p):
    return t.lower() == p.lower()

def count_invalid(pred):
    return sum([1 if p == INVALID_ANS else 0 for p in pred]) / len(pred)

def count_invalid_0(pred):
    return sum([1 if p == INVALID_ANS + "0" else 0 for p in pred]) / len(pred)

def count_invalid_2(pred):
    return sum([1 if p == INVALID_ANS + "2" else 0 for p in pred]) / len(pred)

def accuracy(targ, pred):
    return sum([1 if is_correct(t, p) else 0 for t, p in zip(targ, pred)]) / len(targ)

def precision(targ, pred):
    if sum([1 if p == 'P' else 0 for p in pred]) == 0:
        return None
    return sum([1 if t == p and t == 'P' else 0 for t, p in zip(targ, pred)]) / sum([1 if p == 'P' else 0 for p in pred])

def recall(targ, pred):
    if sum([1 if t == 'P' else 0 for t in targ]) == 0:
        return None
    return sum([1 if t == p and t == 'P' else 0 for t, p in zip(targ, pred)]) / sum([1 if t == 'P' else 0 for t in targ])

def f1score(targ, pred):
    pr = precision(targ, pred)
    rc = recall(targ, pred)
    if pr == None or rc == None:
        return None
    return 1/(1/pr + 1/rc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument("-r", "--result-path", type=str, help="Checkpoint path")
    parser.add_argument("-d", "--dataset-name", type=str, choices=['agnews', 'mr', 'r8', 'sst2'])
    args = parser.parse_args()
    
    print(f"\n========= {args.result_path} =========")
    
    if not os.path.exists(args.result_path):
        print("There is no file")
    
    with open(args.result_path, "r") as file:
        data = json.load(file)
        
    assert check_result_length(data, args.dataset_name)
        
    answers = [example['answer'] for example in data]
    predictions = [extract_answer(args.dataset_name, example['prediction']) for example in data]
    
    new_answers = np.array(answers)
    new_predictions = np.array(predictions)
    
    print("Accuracy : ", accuracy(new_answers, new_predictions))
    print("Number of INVALID ANSWER : ", count_invalid(new_predictions))
    print("Number of INVALID-0 ANSWER : ", count_invalid_0(new_predictions))
    print("Number of INVALID-2 ANSWER : ", count_invalid_2(new_predictions))
    # print("Precision : ", precision(new_answers, new_predictions))
    # print("Recall : ", recall(new_answers, new_predictions))
    # print("F1-score : ", f1score(new_answers, new_predictions))