import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from matplotlib import colors

# Set the path to the ARC dataset
path = ('/Users/vitsiozo/Desktop/MSc AI/Modules/Project/ARC/datasets/2020 dataset')
train_path = os.path.join(path, 'training')
eval_path = os.path.join(path, 'evaluation')
test_path = os.path.join(path, 'test')

# Load the tasks from the training, evaluation and test sets
train_tasks = sorted(os.listdir(train_path))
eval_tasks = sorted(os.listdir(eval_path))
test_tasks = sorted(os.listdir(test_path))

#print ("Number of tasks in the training set: ", len(train_tasks))
#print ("Number of tasks in the evaluation set: ", len(eval_tasks))
#print ("Number of tasks in the test set: ", len(test_tasks))

# Load the first task from the training set
training_task_file = str(os.path.join(train_path,  train_tasks[0]))
#print ("Training task file: ", training_task_file)

with open (training_task_file, 'r') as f:
    task = json.load(f)

n_examples = len(task['train'])
n_tests = len(task['test'])

print ("Task file: ", training_task_file)
print ("Number of training examples:", n_examples)
print ("Number of test examples:", n_tests)

print(task['train'][1]['input'])
print(task['train'][1]['output'])


cmap = colors.ListedColormap(
    ['#000000', '#3399FF','#FF3333','#66CC33','#FFCC00',
     '#999999', '#CC3399', '#FF9933', '#99CCFF', '#990033'])
norm = colors.Normalize(vmin=0, vmax=9)

#plotting the training task and the test task.
def plot_task(task):
    n = len(task["train"]) + len(task["test"])
    fig, axs = plt.subplots(2, n, figsize=(4*n,8), dpi=50)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig_num = 0
    for i, t in enumerate(task["train"]):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Train-{i} in')
        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f'Train-{i} out')
        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))
        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))
        fig_num += 1
    for i, t in enumerate(task["test"]):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Test-{i} in')
        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f'Test-{i} out')
        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))
        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))
        fig_num += 1
    
    plt.tight_layout()
    plt.show()


for i, json_path in enumerate(train_tasks[120:123]):
    task_file = os.path.join(train_path, json_path)
    with open(task_file, 'r') as f:
        task = json.load(f)

    print(f"{i:03d}", task_file)
    plot_task(task)
