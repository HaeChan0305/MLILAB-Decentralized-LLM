import os
import json
import argparse
import jsonlines
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.io import arff


INVALID_ANS = "[invalid]"

def extract_answer(completion):
    try:
        ret = completion.split('assistant\n')[-1]
    except:
        # print(f"WARNING : {completion}")
        ret = completion
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument("-c", "--checkpoint", type=int, help="Checkpoint path")
    parser.add_argument("-c1", "--client1", type=int)
    parser.add_argument("-c2", "--client2", type=int)
    parser.add_argument("-r", "--round", type=int)

    args = parser.parse_args()

    test = []
    
    if args.round == 1:
        prev_test_path = "./pollen_classificatin/pollen_test.jsonl"
        prev_result_path = f"result.json"
    else:
        prev_test_path = f"output_qwen_4_{args.client1}/checkpoint-{args.checkpoint}/pollen_test_round_{args.round - 1}.jsonl"
        prev_result_path = f"result_round_{args.round - 1}.json"
        
    with jsonlines.open(prev_test_path) as file:
        for line in file:
            test.append(line)


    with open(f"/workspace/output_qwen_4_{args.client1}/checkpoint-{args.checkpoint}/{prev_result_path}", "r") as file:
        client_1 = json.load(file)
        client_1_answers = [extract_answer(d['prediction']) for d in client_1]
        
    with open(f"/workspace/output_qwen_4_{args.client2}/checkpoint-{args.checkpoint}/{prev_result_path}", "r") as file:
        client_2 = json.load(file)
        client_2_answers = [extract_answer(d['prediction']) for d in client_2]

    assert len(test) == len(client_1_answers)
    assert len(test) == len(client_2_answers)

    new_test = []
    for t, c1, c2 in zip(test, client_1_answers, client_2_answers):
        t['messages'].insert(-1, {"role" : "assistant",
                                "content" : c1})
        
        t['messages'].insert(-1, {"role" : "user",
                                  "content" : f"This is the recent/updated answer from another agent: {c2}. Use this answer carefully as additional advice, can you provide an updated answer? Make sure to state your answer at the end of the response."})
        
        new_test.append(t)


    with open(f"/workspace/output_qwen_4_{args.client1}/checkpoint-{args.checkpoint}/pollen_test_round_{args.round}.jsonl" , encoding= "utf-8",mode="w") as file: 
        for i in new_test: file.write(json.dumps(i) + "\n")