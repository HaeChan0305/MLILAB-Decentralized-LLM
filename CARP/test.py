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
CARP_PROMPT_1 = """Classify the topic of the input sentence as Sports, World, Science/Technology, or Business.
First, list CLUES (i.e., keywords, phrases, contextual information, semantic relations, semantic meaning, tones, references) that support to classify topic of input.
Second, deduce the diagnostic REASONING process from premises (i.e., clues, input) that supports to determine the topic of the INPUT (Limit the number of words to 130).
Third, based on clues, reasoning, and input, determine the overall TOPIC of the INPUT as either Sports, World, Science/Technology, Business.

The OUTPUT should follow the below JSON format :

{{
    "CLUES" : List of words,
    "REASONING" : String,
    "TOPIC" : String (Only Sports, World, Science/Technology or Business)
}}

INPUT : {0}
OUTPUT :"""

# A prompt for MR and SST2 with CARP
CARP_PROMPT_2 = """Classify the sentiment of the input sentence as Positive or Negative.
First, list CLUES (i.e., keywords, phrases, contextual information, semantic relations, semantic meaning, tones, references) that support to classify sentiment of input.
Second, deduce the diagnostic REASONING process from premises (i.e., clues, input) that supports to determine the sentiment of the INPUT (Limit the number of words to 130).
Third, based on clues, reasoning, and input, determine the overall SENTIMENT of the INPUT as either Positive or Negative.

The OUTPUT should follow the below JSON format :

{{
    "CLUES" : List of words,
    "REASONING" : String,
    "SENTIMENT" : String (Only Positive or Negative)
}}

INPUT : {0}
OUTPUT :"""

# A prompt for r8 with CARP
CARP_PROMPT_3 = """Classify the topic of the input sentence as Grain, Earnings and Earnings Forecasts, Interest Rates, Money/Foreign Exchange, Acquisitions, Crude Oil, Shipping, or Trade.
First, list CLUES (i.e., keywords, phrases, contextual information, semantic relations, semantic meaning, tones, references) that support to classify topic of input.
Second, deduce the diagnostic REASONING process from premises (i.e., clues, input) that supports to determine the topic of the INPUT (Limit the number of words to 130).
Third, based on clues, reasoning, and input, determine the overall TOPIC of the INPUT as either Grain, Earnings and Earnings Forecasts, Interest Rates, Money/Foreign Exchange, Acquisitions, Crude Oil, Shipping, or Trade.
    
The OUTPUT should follow the below JSON format :

{{
    "CLUES" : List of words,
    "REASONING" : String,
    "TOPIC" : String (Only Grain, Earnings and Earnings Forecasts, Interest Rates, Money/Foreign Exchange, Acquisitions, Crude Oil, Shipping, or Trade)
}}

INPUT : {0}
OUTPUT :"""

# A prompt for agnews with CARP
PROMPT_1 = """Classify the topic of the input sentence as Sports, World, Science/Technology, or Business.

The OUTPUT should follow the below JSON format :

{{
    "TOPIC" : String (Only Sports, World, Science/Technology or Business)
}}

INPUT : {0}
OUTPUT :"""


# A prompt for MR and SST2 with CARP
PROMPT_2 = """Classify the sentiment of the input sentence as Positive or Negative.

The OUTPUT should follow the below JSON format :

{{
    "SENTIMENT" : String (Only Positive or Negative)
}}

INPUT : {0}
OUTPUT :"""

# A prompt for r8 with CARP
PROMPT_3 = """Classify the topic of the input sentence as Grain, Earnings and Earnings Forecasts, Interest Rates, Money/Foreign Exchange, Acquisitions, Crude Oil, Shipping, or Trade.

The OUTPUT should follow the below JSON format :

{{
    "TOPIC" : String (Only Grain, Earnings and Earnings Forecasts, Interest Rates, Money/Foreign Exchange, Acquisitions, Crude Oil, Shipping, or Trade)
}}

INPUT : {0}
OUTPUT :"""

# # For exp 5-0-1 / exp 5-0-2
# def processing_text(tokenizer, prompt):
#     messages = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": prompt}
#     ]

#     return tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )

# For exp 5-0-3
def processing_text(tokenizer, messages):
    # assert 'assistant' not in [m['role'] for m in messages]
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
    parser.add_argument("-d", "--dataset-path", type=str)
    parser.add_argument("-o", "--sample-output-file", type=str)
    parser.add_argument("-b", "--batch-size", type=int, default=1)
    # parser.add_argument("-e", "--use-carp", type=str2bool)

    args = parser.parse_args()
    
    # if args.use_carp == True:
    #     if args.dataset == 'agnews': prompt = CARP_PROMPT_1
    #     elif args.dataset == 'mr': prompt = CARP_PROMPT_2
    #     elif args.dataset == 'r8': prompt = CARP_PROMPT_3
    #     elif args.dataset == 'sst2': prompt = CARP_PROMPT_2
    #     else: assert 0        
    # else:
    #     if args.dataset == 'agnews': prompt = PROMPT_1
    #     elif args.dataset == 'mr': prompt = PROMPT_2
    #     elif args.dataset == 'r8': prompt = PROMPT_3
    #     elif args.dataset == 'sst2': prompt = PROMPT_2
    #     else: assert 0
            
    # import pdb; pdb.set_trace()
    # test = pd.read_csv(f"./data/{args.dataset}_test.csv")
    test = []
    with jsonlines.open(args.dataset_path) as file:
        for line in file:
            test.append(line)
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
        
        texts = [processing_text(tokenizer, msg[:-1]) for msg in batch]
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
        
        result += [{
                    'prompt': texts[j],
                    'answer': batch[j][-1]['content'],
                    'prediction': predictions[j]
                   }
                   for j in range(len(predictions))]
        
        with open(os.path.join(args.sample_output_file), "w") as file:
            json.dump(result, file)