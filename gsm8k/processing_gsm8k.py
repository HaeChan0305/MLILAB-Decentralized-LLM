import os
import json
from datasets import load_dataset

SPLIT = "test"

dataset = load_dataset("openai/gsm8k", "main")
processed_dataset = []
for i, sample in enumerate(dataset[SPLIT]):
    processed_dataset.append(
        {
            "type": "chatml", 
            "messages": 
                [
                    {"role": "system", "content": "You are a helpful assistant."}, 
                    {"role": "user", "content": sample['question']}, 
                    {"role": "assistant", "content": sample['answer']}
                ], 
            "source": "unknown"
        }
    )
    

with open(f"qwen2_gsm8k_{SPLIT}.jsonl" , encoding= "utf-8",mode="w") as file: 
	for i in processed_dataset: file.write(json.dumps(i) + "\n")
