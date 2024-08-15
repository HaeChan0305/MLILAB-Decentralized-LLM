import os
import argparse
import matplotlib.pyplot as plt

PATTERN_1 = "Accuracy :  "
PATTERN_2 = "checkpoint-"
STEPS_PER_EPOCH_1 = {
    'agnews': None,
    'mr': 283,
    'r8': 171,
    'sst2': None,
}
STEPS_PER_EPOCH_2 = {
    'agnews': None,
    'mr': 141,
    'r8': 85,
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

def extract_epoch(path, dataset_name, steps_per_epoch):
    with open(path, 'r') as file:
        lines = file.readlines()
    
    ret = []
    for line in lines:
        if PATTERN_2 in line:
            line = line.split(PATTERN_2)[1].split('/')[0]
            ret.append(int(line) / steps_per_epoch[dataset_name])    
    return ret



if __name__ == "__main__":
    dataset_name = "r8"
    
    save_path = f"./plot_{dataset_name}.png"
    
    accuracys_5_0_3 = extract_accuracy(f"output_5_0_3/{dataset_name}/result.txt")
    epochs_5_0_3 = extract_epoch(f"output_5_0_3/{dataset_name}/result.txt", dataset_name, STEPS_PER_EPOCH_1)
    
    accuracys_5_1_1 = extract_accuracy(f"output_5_1_1/{dataset_name}/result.txt")
    epochs_5_1_1 = extract_epoch(f"output_5_1_1/{dataset_name}/result.txt", dataset_name, STEPS_PER_EPOCH_2)
    
    accuracys_5_1_2 = extract_accuracy(f"output_5_1_2/{dataset_name}/result.txt")
    epochs_5_1_2 = extract_epoch(f"output_5_1_2/{dataset_name}/result.txt", dataset_name, STEPS_PER_EPOCH_2)
    
    # Plot the data
    plt.plot(epochs_5_0_3, accuracys_5_0_3, label='Baseline (Exp 5-0-3)')
    plt.plot(epochs_5_1_1, accuracys_5_1_1, label='Client 1  (Exp 5-1-1)')
    plt.plot(epochs_5_1_2, accuracys_5_1_2, label='Client 2  (Exp 5-1-2)')

    # Add labels and title
    plt.xlabel('Epoches')
    plt.ylabel('Accuracy')
    plt.title(f'IID Clients\' Accuracy : {dataset_name}')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()
    plt.savefig(save_path)