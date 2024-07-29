import pandas as pd
from scipy.io import arff
from pycaret.datasets import get_data
from sklearn.model_selection import train_test_split
from pycaret.regression import *


dataset = arff.loadarff('./pollen.arff')
dataset = pd.DataFrame(dataset[0])
dataset = dataset.drop(columns=['OBSERVATION_NUMBER'])

train, test = train_test_split(dataset, test_size=0.15, random_state=42)

reg = setup(
    data=train, 
    test_data=test,
    target='DENSITY', 
    numeric_features=['RIDGE', 'NUB', 'CRACK', 'WEIGHT']
    )
best = compare_models()
