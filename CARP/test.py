import os
import re
import torch
import argparse
import json
import jsonlines
import numpy as np
import datasets
import pandas as pd
from tqdm import tqdm
from datasets import load_from_disk, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

# A prompt for agnews with CARP
CARP_PROMPT_1 = """Classify the topic of the input sentence as (1) Sports, (2) World, (3) Science/Technology, or (4) Business.
First, list CLUES (i.e., keywords, phrases, contextual information, semantic relations, semantic meaning, tones, references) that support to classify topic of input.
Second, deduce the diagnostic REASONING process from premises (i.e., clues, input) that supports to determine the topic of the INPUT (Limit the number of words to 130).
Third, based on clues, reasoning, and input, determine the overall TOPIC of the INPUT as either : 
    
    (1) Sports
    (2) World
    (3) Science/Technology
    (4) Business

The OUTPUT should follow the below JSON format :

{{
    "CLUES" : List of words,
    "REASONING" : String,
    "TOPIC" : Int (Only 1, 2, 3 and 4 are possible)
}}

INPUT : {0}
OUTPUT :"""

# A prompt for MR and SST2 with CARP
CARP_PROMPT_2 = """Classify the sentiment of the input sentence as (1) Positive or (2) Negative.
First, list CLUES (i.e., keywords, phrases, contextual information, semantic relations, semantic meaning, tones, references) that support to classify sentiment of input.
Second, deduce the diagnostic REASONING process from premises (i.e., clues, input) that supports to determine the sentiment of the INPUT (Limit the number of words to 130).
Third, based on clues, reasoning, and input, determine the overall SENTIMENT of the INPUT as either :
    
    (1) Positive
    (2) Negative

The OUTPUT should follow the below JSON format :

{{
    "CLUES" : List of words,
    "REASONING" : String,
    "SENTIMENT" : Int (Only 1 and 2 are possible)
}}

INPUT : {0}
OUTPUT :"""

# A prompt for r8 with CARP
CARP_PROMPT_3 = """Classify the topic of the input sentence as (1) Grain, (2) Earnings and Earnings Forecasts, (3) Interest Rates, (4) Money/Foreign Exchange, (5) Acquisitions, (6) Crude Oil, (7) Shipping, or (8) Trade.
First, list CLUES (i.e., keywords, phrases, contextual information, semantic relations, semantic meaning, tones, references) that support to classify topic of input.
Second, deduce the diagnostic REASONING process from premises (i.e., clues, input) that supports to determine the topic of the INPUT (Limit the number of words to 130).
Third, based on clues, reasoning, and input, determine the overall TOPIC of the INPUT as either :
    
    (1) Grain
    (2) Earnings and Earnings Forecasts
    (3) Interest Rates
    (4) Money/Foreign Exchange
    (5) Acquisitions
    (6) Crude Oil
    (7) Shipping
    (8) Trade

The OUTPUT should follow the below JSON format :

{{
    "CLUES" : List of words,
    "REASONING" : String,
    "TOPIC" : Int (Only 1, 2, 3, 4, 5, 6, 7 and 8 are possible)
}}

INPUT : {0}
OUTPUT :"""

# A prompt for agnews with CARP
PROMPT_1 = """Classify the topic of the input sentence as :
    
    (1) Sports
    (2) World
    (3) Science/Technology
    (4) Business

The OUTPUT should follow the below JSON format :

{{
    "TOPIC" : Int (Only 1, 2, 3 and 4 are possible)
}}

INPUT : {0}
OUTPUT :"""


# A prompt for MR and SST2 with CARP
PROMPT_2 = """Classify the sentiment of the input sentence as :
    
    (1) Positive
    (2) Negative

The OUTPUT should follow the below JSON format :

{{
    "SENTIMENT" : Int (Only 1 and 2 are possible)
}}

INPUT : {0}
OUTPUT :"""

# A prompt for r8 with CARP
PROMPT_3 = """Classify the topic of the input sentence as :
    
    (1) Grain
    (2) Earnings and Earnings Forecasts
    (3) Interest Rates
    (4) Money/Foreign Exchange
    (5) Acquisitions
    (6) Crude Oil
    (7) Shipping
    (8) Trade

The OUTPUT should follow the below JSON format :

{{
    "TOPIC" : Int (Only 1, 2, 3, 4, 5, 6, 7 and 8 are possible)
}}

INPUT : {0}
OUTPUT :"""


def processing_text(tokenizer, prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

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
    parser.add_argument("-c", "--checkpoint-path", type=str, help="Checkpoint path", default="Qwen/Qwen2-1.5B-Instruct")
    parser.add_argument("-d", "--dataset", type=str, choices=['agnews', 'mr', 'r8', 'sst2'])
    parser.add_argument("-o", "--sample-output-file", type=str)
    parser.add_argument("-b", "--batch-size", type=int, default=1)
    parser.add_argument("-e", "--use-carp", type=str2bool)

    args = parser.parse_args()
    
    if args.use_carp == True:
        if args.dataset == 'agnews': prompt = CARP_PROMPT_1
        elif args.dataset == 'mr': prompt = CARP_PROMPT_2
        elif args.dataset == 'r8': prompt = CARP_PROMPT_3
        elif args.dataset == 'sst2': prompt = CARP_PROMPT_2
        else: assert 0        
    else:
        if args.dataset == 'agnews': prompt = PROMPT_1
        elif args.dataset == 'mr': prompt = PROMPT_2
        elif args.dataset == 'r8': prompt = PROMPT_3
        elif args.dataset == 'sst2': prompt = PROMPT_2
        else: assert 0
            
    # import pdb; pdb.set_trace()
    test = pd.read_csv(f"./data/{args.dataset}_test.csv")

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
        
        batch = batch.reset_index()
        texts = [processing_text(tokenizer, prompt.format(row['sentence'])) for _, row in batch.iterrows()]
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
        
        # import pdb; pdb.set_trace()
        result += [{
                    'prompt': texts[j],
                    'sentence': batch['sentence'][j],
                    'answer': batch['label'][j],
                    'prediction': predictions[j]
                   }
                   for j in range(len(predictions))]
            
        with open(os.path.join(args.sample_output_file), "w") as file:
            json.dump(result, file)