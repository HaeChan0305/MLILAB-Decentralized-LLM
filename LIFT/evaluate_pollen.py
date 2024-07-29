import os
import re
import torch
import argparse
import json
import jsonlines
import numpy as np
import datasets
from tqdm import tqdm
from datasets import load_from_disk, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

TEMPLATE = "{% for message in messages[:2] %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{{ '<|im_end|>\n' }}{% endfor %}"

def processing_text(tokenizer, messages):

    return tokenizer.apply_chat_template(
        messages,
        chat_template=TEMPLATE,
        tokenize=False,
        add_generation_prompt=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument("-c", "--checkpoint-path", type=str, help="Checkpoint path", default="Qwen/Qwen2-1.5B-Instruct")
    # parser.add_argument("-f", "--sample-input-file", type=str, default=None)
    parser.add_argument("-o", "--sample-output-file", type=str)
    parser.add_argument("-b", "--batch-size", type=int, default=1)

    args = parser.parse_args()

    test = []
    with jsonlines.open("pollen_test.jsonl") as file:
        for line in file:
            test.append(line)
    # import pdb;pdb.set_trace()
    test = [example["messages"] for example in test]

    device = "cuda" # the device to load the model onto

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path,
                                              padding_side='left')

    B = args.batch_size
    for i in tqdm(range(len(test)//B + 1)):
        if B * (i+1) < len(test):
            batch = test[B * i : B * (i+1)]
        else:
            batch = test[B * i : ]
        
        texts = [processing_text(tokenizer, msg) for msg in batch]
        # import pdb; pdb.set_trace()
        model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
        
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )        
        generated_ids =[output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids,  generated_ids)]

        predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        

        if i != 0:
            with open(os.path.join(args.sample_output_file), "r") as file:
                result = json.load(file)
        else:
            result = []
        
        result += [{'question': batch[j][1]['content'],
                    'answer': batch[j][2]['content'],
                    'prediction': predictions[j]
                    }
                   for j in range(len(predictions))]
            
        with open(os.path.join(args.sample_output_file), "w") as file:
            json.dump(result, file)