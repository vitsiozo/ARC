import numpy as np
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

# Load the first task from the training set
training_task_file = str(os.path.join(train_path,  "d631b094.json"))

with open(training_task_file, 'r') as f:
    task = json.load(f)

cmap = colors.ListedColormap(
    ['#000000', '#3399FF','#FF3333','#66CC33','#FFCC00',
     '#999999', '#CC3399', '#FF9933', '#99CCFF', '#990033'])
norm = colors.Normalize(vmin=0, vmax=9)

def plot_task(task):
    n = len(task["train"]) + len(task["test"])
    fig, axs = plt.subplots(2, n, figsize=(4*n, 8), dpi=50)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig_num = 0
    
    for i, t in enumerate(task["train"]):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        
        # Plot input
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Train-{i} in')
        axs[0][fig_num].grid(True, color='gray', linestyle='-', linewidth=0.5)
        axs[0][fig_num].set_xticks([])
        axs[0][fig_num].set_yticks([])
        axs[0][fig_num].set_xticks(np.arange(-0.5, t_in.shape[1], 1), minor=True)
        axs[0][fig_num].set_yticks(np.arange(-0.5, t_in.shape[0], 1), minor=True)
        axs[0][fig_num].grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        
        # Plot output
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f'Train-{i} out')
        axs[1][fig_num].grid(True, color='gray', linestyle='-', linewidth=0.5)
        axs[1][fig_num].set_xticks([])
        axs[1][fig_num].set_yticks([])
        axs[1][fig_num].set_xticks(np.arange(-0.5, t_out.shape[1], 1), minor=True)
        axs[1][fig_num].set_yticks(np.arange(-0.5, t_out.shape[0], 1), minor=True)
        axs[1][fig_num].grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        
        fig_num += 1

    for i, t in enumerate(task["test"]):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        
        # Plot input
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Test-{i} in')
        axs[0][fig_num].grid(True, color='gray', linestyle='-', linewidth=0.5)
        axs[0][fig_num].set_xticks([])
        axs[0][fig_num].set_yticks([])
        axs[0][fig_num].set_xticks(np.arange(-0.5, t_in.shape[1], 1), minor=True)
        axs[0][fig_num].set_yticks(np.arange(-0.5, t_in.shape[0], 1), minor=True)
        axs[0][fig_num].grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        
        # Plot output
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f'Test-{i} out')
        axs[1][fig_num].grid(True, color='gray', linestyle='-', linewidth=0.5)
        axs[1][fig_num].set_xticks([])
        axs[1][fig_num].set_yticks([])
        axs[1][fig_num].set_xticks(np.arange(-0.5, t_out.shape[1], 1), minor=True)
        axs[1][fig_num].set_yticks(np.arange(-0.5, t_out.shape[0], 1), minor=True)
        axs[1][fig_num].grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        
        fig_num += 1
    
    plt.tight_layout()
    plt.show()

plot_task(task)
