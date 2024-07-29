import os
import re
import json
import argparse
import numpy as np

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
    
def extract_answer(completion):
    pattern = r"-?\d+\.\d+"
    try:
        matches = re.findall(pattern, completion)
        ret = float(matches[-1])
    except:
        ret = INVALID_ANS
    return ret

def mae(t, p):
    return (np.abs(t - p)).mean()

def mse(t, p):
    return ((t - p) ** 2).mean()

def rmse(t, p):
    return np.sqrt(((t - p) ** 2).mean())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument("-c1", "--client1-path", type=str, help="Checkpoint path")
    parser.add_argument("-c2", "--client2-path", type=str, help="Checkpoint path")
    args = parser.parse_args()
    
    print(f"\n========= {args.client1_path} =========")
    print(f"========= {args.client2_path} =========")
    
    if not os.path.exists(args.client1_path):
        print("There is no file")
    
    if not os.path.exists(args.client2_path):
        print("There is no file")
    
    with open(args.client1_path, "r") as file:
        data1 = json.load(file)
        
    with open(args.client2_path, "r") as file:
        data2 = json.load(file)
    
    if len(data1) != 578:
        print(f"data length is {len(data)}, not 578.")
        
    if len(data2) != 578:
        print(f"data length is {len(data)}, not 578.")
    
    answers1 = [float(example['answer']) for example in data1]
    predictions1 = [extract_answer(example['prediction']) for example in data1]
    
    answers2 = [float(example['answer']) for example in data2]
    predictions2 = [extract_answer(example['prediction']) for example in data2]
    
    assert answers1 == answers2
    assert predictions1 != predictions2
    
    new_answers = []
    new_predictions = []
    for a, p1, p2 in zip(answers1, predictions1, predictions2):
        if not p1 == INVALID_ANS and not p2 == INVALID_ANS:
            new_answers.append(a)
            new_predictions.append((p1+p2)/2)
    
    new_answers = np.array(new_answers)
    new_predictions = np.array(new_predictions)
    
    print("MAE : ", mae(new_answers, new_predictions))
    print("MSE : ", mse(new_answers, new_predictions))
    print("RMSE : ", rmse(new_answers, new_predictions))
    
    
    
    
    
    