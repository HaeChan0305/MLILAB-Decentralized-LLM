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

def plot_exp_5_2_1():
    dataset_name = "mr"
    client = 2
    
    save_path = f"./plot_debate_{dataset_name}_client2.png"
    
    accuracyss = []
    epochss = []
    for round in range(9):
        if round == 0:
            result_path = "result.txt"
            accuracys = extract_accuracy(f"output_5_1_{client}/{dataset_name}/{result_path}")[1:]
            epochs = extract_epoch(f"output_5_1_{client}/{dataset_name}/{result_path}", dataset_name, STEPS_PER_EPOCH_2)[1:]
        else:
            result_path = f"result_round_{round}.txt"
            accuracys = extract_accuracy(f"output_5_1_{client}/{dataset_name}/{result_path}")
            epochs = extract_epoch(f"output_5_1_{client}/{dataset_name}/{result_path}", dataset_name, STEPS_PER_EPOCH_2)

        accuracyss.append(accuracys)
        epochss.append(epochs)

    # Plot the data
    for i, (accuracys, epochs) in enumerate(zip(accuracyss, epochss)):
        plt.plot(epochs, accuracys, label=f'Client {client} - Round {i}')
    

    # Add labels and title
    plt.xlabel('Epoches')
    plt.ylabel('Accuracy')
    plt.title(f'IID Client {client} Debate Accuracy : {dataset_name}')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()
    plt.savefig(save_path)


if __name__ == "__main__":
    dataset_name = "mr"
    client = 2
    
    save_path = f"./plot_debate_{dataset_name}_client{client}.png"
    
    accuracys = extract_accuracy(f"output_5_2_{client}/{dataset_name}/result.txt")
    rounds = range(len(accuracys))
    
    # Plot the data
    plt.plot(rounds, accuracys, label=f'Client {client}')
    
    # Add labels and title
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.title(f'IID Client {client} Debate Accuracy : {dataset_name}')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()
    plt.savefig(save_path)