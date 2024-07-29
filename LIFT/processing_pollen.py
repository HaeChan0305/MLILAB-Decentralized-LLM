import os
import json
import jsonlines
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.io import arff
from sklearn.utils import shuffle

# For regression
# Q_TEMPLATE = "The first three variables are the lengths of geometric features observed sampled pollen grains - in the x, y, and z dimensions: a ridge={0} along x, a nub={1} in the y direction, and a crack={2} in along the z dimension. The fourth variable is pollen grain weight={3}. What is the density of it?"
# A_TEMPLATE = "{0}"

# For classification
Q_TEMPLATE = """It converts the numeric target feature to a two-class nominal target feature by computing the mean and classifying all instances with a lower target value as positive ('P') and all others as negative ('N'). The first three variables are the lengths of geometric features observed sampled pollen grains - in the x, y, and z dimensions: a ridge={0} along x, a nub={1} in the y direction, and a crack={2} in along the z dimension. The fourth variable is pollen grain weight={3}, and the fifth is density={4}. What is the target value? You should answer between 'P' and 'N'."""
A_TEMPLATE = "{0}"


dataset = arff.loadarff('./pollen_classification/pollen_classification.arff')
dataset = pd.DataFrame(dataset[0])
dataset['binaryClass'] = ['P' if c == b'P' else 'N' for c in dataset['binaryClass']]


dataset_P = dataset.where(dataset['binaryClass']=='P').dropna()
dataset_N = dataset.where(dataset['binaryClass']=='N').dropna()

train_P, test_P = train_test_split(dataset_P, test_size=0.15, random_state=42)
train_N, test_N = train_test_split(dataset_N, test_size=0.15, random_state=42)

train_P_1 = train_P.sample(frac = 0.9)
train_P_2 = train_P.drop(train_P_1.index)
train_N_1 = train_N.sample(frac = 0.1)
train_N_2 = train_N.drop(train_N_1.index)

train_1 = shuffle(pd.concat([train_P_1, train_N_1]))
train_2 = shuffle(pd.concat([train_P_2, train_N_2]))
train = pd.concat([train_1, train_2])
test = shuffle(pd.concat([test_P, test_N]))


for SPLIT in ['train', 'test', 'train_1', 'train_2']:
    if SPLIT == 'train': dataset = train 
    elif SPLIT == 'train_1': dataset = train_1
    elif SPLIT == 'train_2': dataset = train_2 
    elif SPLIT == 'test': dataset = test 
    else: assert 0
    
    processed_dataset = []
    for _, row in dataset.iterrows():
        processed_dataset.append(
            {
                "type": "chatml", 
                "messages": 
                    [
                        {"role": "system", "content": "You are a helpful assistant."}, 
                        {"role": "user", "content": Q_TEMPLATE.format(row['RIDGE'], row['NUB'], row['CRACK'], row['WEIGHT'], row['DENSITY'])}, 
                        {"role": "assistant", "content": A_TEMPLATE.format(row['binaryClass'])}
                    ], 
                "source": "unknown"
            }
        )
        
    with open(f"./pollen_classification/pollen_{SPLIT}.jsonl" , encoding= "utf-8",mode="w") as file: 
        for i in processed_dataset: file.write(json.dumps(i) + "\n")
    
# train = []
# with jsonlines.open("pollen_train.jsonl") as file:
#     for line in file:
#         train.append(line)


# N = len(train)
# with open(f"pollen_train_1.jsonl" , encoding= "utf-8",mode="w") as file: 
#     for i in train[:N//2]: file.write(json.dumps(i) + "\n")
    
# with open(f"pollen_train_2.jsonl" , encoding= "utf-8",mode="w") as file: 
#     for i in train[N//2:]: file.write(json.dumps(i) + "\n")

