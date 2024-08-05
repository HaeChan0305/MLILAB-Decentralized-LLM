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
    device = "cuda" # the device to load the model onto

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-1.5B-Instruct",
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct",
                                              padding_side='left')

    texts = """This is an overall sentiment classifier for movie reviews. Classify the overall SENTIMENT of the INPUT as Positive or Negative. The output should be a single word.\nINPUT: press the delete key.\nSENTIMENT:"""
    
    model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
        
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )        
    generated_ids =[output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids,  generated_ids)]

    predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    print("#####")
    print(predictions[0])