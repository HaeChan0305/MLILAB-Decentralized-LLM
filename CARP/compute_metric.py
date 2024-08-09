import os
import re
import json
import argparse
import numpy as np
from post_processing import reconstruct

INVALID_ANS = "[invalid]"

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

def check_result_length(data, dataset_name):
    truth_table = {
        "agnews" : 7600,
        "mr" : 1600,
        "r8" : 2189,
        "sst2" : 872
    }
    
    if len(data) != truth_table[dataset_name]:
        print(f"{dataset_name} length is {len(data)}. It should be {truth_table[dataset_name]}.")
        return False
    else:
        return True

def extract_answer_carp(dataset_name, completion):
    for i in range(2):
        try:
            json_object = json.loads(completion)
            
            if json_object.keys() != set(['CLUES', 'REASONING', KEYWORDS[dataset_name]]):
                return INVALID_ANS + "0"
            if type(json_object['CLUES']) != list:
                return INVALID_ANS + "0"
            if len(json_object['CLUES']) > 0 and type(json_object['CLUES'][0]) != str:
                return INVALID_ANS + "0"
            if type(json_object['REASONING']) != str:
                return INVALID_ANS + "0"
            if type(json_object[KEYWORDS[dataset_name]]) != str:
                return INVALID_ANS + "0"
            if json_object[KEYWORDS[dataset_name]].lower() not in [c.lower() for c in CHOICES[dataset_name]]:
                return INVALID_ANS + "1"
            return json_object[KEYWORDS[dataset_name]]

        except Exception as e:
            if i == 1:
                return INVALID_ANS + "0"
            
            completion = reconstruct(completion)
            if completion == None:
                return INVALID_ANS + "0"


def extract_answer_no_carp(dataset_name, completion):
    for i in range(2):
        try:
            json_object = json.loads(completion)
            
            if json_object.keys() != set([KEYWORDS[dataset_name]]):
                return INVALID_ANS + "0"
            if type(json_object[KEYWORDS[dataset_name]]) != str:
                return INVALID_ANS + "0"
            if json_object[KEYWORDS[dataset_name]].lower() not in [c.lower() for c in CHOICES[dataset_name]]:
                return INVALID_ANS + "1"
            return json_object[KEYWORDS[dataset_name]]

        except Exception as e:
            if i == 1:
                return INVALID_ANS + "0"
            
            new_completion = None
            for choice in CHOICES[dataset_name]:
                if choice in completion:
                    splits = completion.split(choice)
                    new_completion = splits[0] + '"' + choice + '"' + splits[1]
                    # import pdb; pdb.set_trace()
                    break

            if new_completion == None:
                return INVALID_ANS + "0"
            else:
                completion = new_completion
        
def extract_answer_base(dataset_name, completion):
    ret = None
    for choice in CHOICES[dataset_name]:
        if choice in completion:
            if ret == None:
                ret = choice
            else:
                return INVALID_ANS + "0"
    
    if ret == None:
        return INVALID_ANS + "1"
    else:
        return ret

def is_correct(t, p):
    return t.lower() == p.lower()

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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument("-r", "--result-path", type=str, help="Checkpoint path")
    parser.add_argument("-d", "--dataset-name", type=str, choices=['agnews', 'mr', 'r8', 'sst2'])
    # parser.add_argument("-e", "--use-carp", type=str2bool)
    args = parser.parse_args()
    
    print(f"\n======== {args.result_path} ========")
    
    if not os.path.exists(args.result_path):
        print("There is no file")
    
    with open(args.result_path, "r") as file:
        data = json.load(file)
        
    assert check_result_length(data, args.dataset_name)
        
    
    answers = [example['answer'] for example in data]
    # # exp_5_0_1 / exp_5_0_2
    # extract_answer = extract_answer_carp if args.use_carp else extract_answer_no_carp
    # predictions = [extract_answer(args.dataset_name, example['prediction']) for example in data]
    
    # # JSON format 깨졌을 때 사용
    # predictions = [extract_answer_base(args.dataset_name, example['prediction']) for example in data] 
    
    # exp_5_0_3
    predictions = [example['prediction'] for example in data] 
    
    print("Accuracy : ", accuracy(answers, predictions))
    
    # print("Number of Total    :", len(predictions))
    # print("Number of Correct  :", sum([1 for t, p in zip(answers, predictions) if is_correct(t, p)]))
    # print("Number of Wrong    :", sum([1 for t, p in zip(answers, predictions) if INVALID_ANS not in p and not is_correct(t, p)]))
    # print("Number of INVALID 0:", sum([1 for p in predictions if p == INVALID_ANS + "0"]))
    # print("Number of INVALID 1:", sum([1 for p in predictions if p == INVALID_ANS + "1"]))
    
    
    
    # wrong_list = [
    #     {"answer" : t, "prediction" : p} 
    #     for t, p in zip(answers, predictions) if INVALID_ANS not in p and not is_correct(t, p)
    # ]
    
    # invalid_list = [example['prediction'] for example, p in zip(data, predictions) if p == INVALID_ANS + "1"]
    # for a in invalid_list:
    #     print(a)
    
    # import pdb; pdb.set_trace()
