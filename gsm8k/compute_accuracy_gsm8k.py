import os
import re
import json
import argparse

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

def extract_answer_hf(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return eval(match_str)
    else:
        return INVALID_ANS


def extract_answer(completion):
    try:
        last_number = re.findall(r"\d+", completion)[-1]
        return eval(last_number)
    except:
        return INVALID_ANS


def is_correct(completion, answer):
    gold = extract_answer_hf(answer)
    assert gold != INVALID_ANS, "No ground truth answer found in the document."
    return extract_answer(completion) == gold

if __name__ == '__main__':
    PATH = "output_qwen_1_3/checkpoint-{0}/result.json"
    CHECKPOINTS = [i * 150 for i in range(1, 4)]
    
    for checkpoint in CHECKPOINTS:
        path = PATH.format(checkpoint)
        print(f"\n========= {path} =========")
        
        if not os.path.exists(path):
            print("There is no file")
            continue
        
        with open(path, "r") as file:
            data = json.load(file)
        
        if len(data) != 1319:
            print(f"data length is {len(data)}, not 1319.")
            continue
        
        accuracy = sum([1 if is_correct(d['prediction'], d['answer']) else 0 for d in data]) / len(data)
        print("accuracy : ", accuracy * 100)
        
        
        
        
        
        