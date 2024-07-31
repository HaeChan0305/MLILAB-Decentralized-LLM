import os
import re
import json
import argparse
import numpy as np

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
    
# def extract_answer(completion):
#     pattern = r"-?\d+\.\d+"
#     try:
#         matches = re.findall(pattern, completion)
#         ret = float(matches[-1])
#     except:
#         ret = INVALID_ANS
#     return ret

def extract_answer(completion):
    try:
        ret = completion.split('\n')[1]
        if not ret in ['P', 'N']:
            ret = INVALID_ANS
    except:
        ret = INVALID_ANS
    return ret

def accuracy(targ, pred):
    return sum([1 if t == p else 0 for t, p in zip(targ, pred)]) / len(targ)

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

def mae(t, p):
    return (np.abs(t - p)).mean()

def mse(t, p):
    return ((t - p) ** 2).mean()

def rmse(t, p):
    return np.sqrt(((t - p) ** 2).mean())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument("-c1", "--client1-path", type=str, help="Checkpoint path")
    # parser.add_argument("-c2", "--client2-path", type=str, help="Checkpoint path")
    args = parser.parse_args()
    
    print(f"\n========= {args.client1_path} =========")
    # print(f"========= {args.client2_path} =========")
    
    if not os.path.exists(args.client1_path):
        print("There is no file")
    
    # if not os.path.exists(args.client2_path):
    #     print("There is no file")
    
    with open(args.client1_path, "r") as file:
        data1 = json.load(file)
        
    # with open(args.client2_path, "r") as file:
    #     data2 = json.load(file)
    
    if len(data1) != 578:
        print(f"data length is {len(data)}, not 578.")
        
    # if len(data2) != 578:
    #     print(f"data length is {len(data)}, not 578.")
    
    # answers1 = [float(example['answer']) for example in data1]
    # import pdb; pdb.set_trace()
    answers1 = [example['answer'] for example in data1]
    predictions1 = [extract_answer(example['prediction']) for example in data1]
    
    # answers2 = [float(example['answer']) for example in data2]
    # predictions2 = [extract_answer(example['prediction']) for example in data2]
    
    # assert answers1 == answers2
    # assert predictions1 != predictions2
    
    new_answers = []
    new_predictions = []
    for a, p1 in zip(answers1, predictions1):
        if not p1 == INVALID_ANS:
            new_answers.append(a)
            new_predictions.append(p1)
    
    new_answers = np.array(new_answers)
    new_predictions = np.array(new_predictions)
    
    # print("MAE : ", mae(new_answers, new_predictions))
    # print("MSE : ", mse(new_answers, new_predictions))
    # print("RMSE : ", rmse(new_answers, new_predictions))
    
    print("Accuracy : ", accuracy(new_answers, new_predictions))
    print("Precision : ", precision(new_answers, new_predictions))
    print("Recall : ", recall(new_answers, new_predictions))
    print("F1-score : ", f1score(new_answers, new_predictions))