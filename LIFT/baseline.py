import pandas as pd
from scipy.io import arff
from pycaret.datasets import get_data
from sklearn.model_selection import train_test_split
# from pycaret.regression import *
from pycaret.classification import ClassificationExperiment

dataset = arff.loadarff('./pollen_classification/pollen_classification.arff')
dataset = pd.DataFrame(dataset[0])
# dataset = dataset.drop(columns=['OBSERVATION_NUMBER'])

train, test = train_test_split(dataset, test_size=0.15, random_state=42)

s = ClassificationExperiment()
s.setup(
    data=train, 
    test_data=test,
    target='binaryClass', 
    numeric_features=['RIDGE', 'NUB', 'CRACK', 'WEIGHT', 'DENSITY'],
    normalize = True,
    log_experiment = True,
    )
# import pdb; pdb.set_trace()

best = s.compare_models()