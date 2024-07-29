import os
import json
import jsonlines
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.io import arff


# Q_TEMPLATE = "The first three variables are the lengths of geometric features observed sampled pollen grains - in the x, y, and z dimensions: a ridge={0} along x, a nub={1} in the y direction, and a crack={2} in along the z dimension. The fourth variable is pollen grain weight={3}. What is the density of it?"
# A_TEMPLATE = "{0}"

# dataset = arff.loadarff('./pollen.arff')
# dataset = pd.DataFrame(dataset[0])
# train, test = train_test_split(dataset, test_size=0.15, random_state=42)

# for SPLIT in ['train', 'test']:
#     dataset = train if SPLIT == 'train' else test
    
#     processed_dataset = []
#     for _, row in dataset.iterrows():
#         processed_dataset.append(
#             {
#                 "type": "chatml", 
#                 "messages": 
#                     [
#                         {"role": "system", "content": "You are a helpful assistant."}, 
#                         {"role": "user", "content": Q_TEMPLATE.format(row['RIDGE'], row['NUB'], row['CRACK'], row['WEIGHT'])}, 
#                         {"role": "assistant", "content": A_TEMPLATE.format(row['DENSITY'])}
#                     ], 
#                 "source": "unknown"
#             }
#         )
        
#     with open(f"pollen_{SPLIT}.jsonl" , encoding= "utf-8",mode="w") as file: 
#         for i in processed_dataset: file.write(json.dumps(i) + "\n")
    
# train = []
# with jsonlines.open("pollen_train.jsonl") as file:
#     for line in file:
#         train.append(line)


# N = len(train)
# with open(f"pollen_train_1.jsonl" , encoding= "utf-8",mode="w") as file: 
#     for i in train[:N//2]: file.write(json.dumps(i) + "\n")
    
# with open(f"pollen_train_2.jsonl" , encoding= "utf-8",mode="w") as file: 
#     for i in train[N//2:]: file.write(json.dumps(i) + "\n")

