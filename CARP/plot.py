import os
import argparse
import matplotlib.pyplot as plt

PATTERN_1 = "Accuracy :  "
PATTERN_2 = "checkpoint-"
STEPS_PER_EPOCH = {
    'agnews': None,
    'mr': 283,
    'r8': 171,
    'sst2': None,
}


def extract_accuracy(path):
    with open(path, 'r') as file:
        lines = file.readlines()
    
    ret = []
    for line in lines:
        if PATTERN_1 in line:
            line = line.replace(PATTERN_1, '')
            line = line.strip()
            ret.append(float(line))    
    return ret

def extract_epoch(path, dataset_name):
    with open(path, 'r') as file:
        lines = file.readlines()
    
    ret = []
    for line in lines:
        if PATTERN_2 in line:
            line = line.split(PATTERN_2)[1].split('/')[0]
            ret.append(int(line) / STEPS_PER_EPOCH[dataset_name])    
    return ret



if __name__ == "__main__":
    dataset_name = "mr"
    
    save_path = f"./plot_{dataset_name}.png"
    
    accuracys_1 = extract_accuracy(f"output_5_0_1/{dataset_name}/result.txt")    
    epochs_1 = extract_epoch(f"output_5_0_1/{dataset_name}/result.txt", dataset_name)
    
    accuracys_2 = extract_accuracy(f"output_5_0_2/{dataset_name}/result.txt")    
    epochs_2 = extract_epoch(f"output_5_0_2/{dataset_name}/result.txt", dataset_name)

    assert epochs_1 == epochs_2
    
    # Plot the data
    plt.plot(epochs_1, accuracys_1, label='No CARP')
    plt.plot(epochs_2, accuracys_2, label='CARP')

    # Add labels and title
    plt.xlabel('Epoches')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy : {dataset_name}')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()
    plt.savefig(save_path)