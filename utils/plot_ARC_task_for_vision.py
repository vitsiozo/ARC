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

# Define the directory where images will be saved
save_directory = '/Users/vitsiozo/Desktop/MSc AI/Modules/Project/ARC/datasets/vision_dataset/'

# Ask user which task to load
while True:
    task_id = input('Enter the ID of the training task you want to plot: ')
    training_task_file = str(os.path.join(train_path,  task_id + ".json"))
    # Check if the file exists
    if os.path.exists(training_task_file):
        break
    else:
        print('File not found. Please enter a valid task ID.')

print(f'Plotting task file: {task_id}.json')

with open(training_task_file, 'r') as f:
    task = json.load(f)

cmap = colors.ListedColormap(
    ['#000000', '#3399FF','#FF3333','#66CC33','#FFCC00',
     '#999999', '#CC3399', '#FF9933', '#99CCFF', '#990033'])
norm = colors.Normalize(vmin=0, vmax=9)

def plot_task(task, task_id, save=False, dpi=300, format="png", figsize=(12, 8)):
    # Set font size and weight globally for the plot
    plt.rcParams.update({'font.size': 14, 'font.weight': 'bold'})  # Adjust this value as needed

    # Generate filename using task_id
    filename = f"{task_id}.png"
    if save_directory:
        filename = os.path.join(save_directory, filename)

    n = len(task["train"]) + len(task["test"])
    fig, axs = plt.subplots(2, n, figsize=(4*n, 8), dpi=50)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    fig_num = 0
    
    for i, t in enumerate(task["train"]):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        
        # Plot input
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Example {i+1} Input', fontsize=16, fontweight='bold')  # Increase font size and set bold
        axs[0][fig_num].grid(True, color='gray', linestyle='-', linewidth=0.5)
        axs[0][fig_num].set_xticks([])
        axs[0][fig_num].set_yticks([])
        axs[0][fig_num].set_xticks(np.arange(-0.5, t_in.shape[1], 1), minor=True)
        axs[0][fig_num].set_yticks(np.arange(-0.5, t_in.shape[0], 1), minor=True)
        axs[0][fig_num].grid(which='minor', color='lightgray', linestyle='-', linewidth=0.5)
        
        # Plot output
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f'Example {i+1} Output', fontsize=16, fontweight='bold')  # Increase font size and set bold
        axs[1][fig_num].grid(True, color='gray', linestyle='-', linewidth=0.5)
        axs[1][fig_num].set_xticks([])
        axs[1][fig_num].set_yticks([])
        axs[1][fig_num].set_xticks(np.arange(-0.5, t_out.shape[1], 1), minor=True)
        axs[1][fig_num].set_yticks(np.arange(-0.5, t_out.shape[0], 1), minor=True)
        axs[1][fig_num].grid(which='minor', color='lightgray', linestyle='-', linewidth=0.5)
        
        fig_num += 1

    for i, t in enumerate(task["test"]):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        
        # Plot input
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Test Input', fontsize=16, fontweight='bold')  # Increase font size and set bold
        axs[0][fig_num].grid(True, color='gray', linestyle='-', linewidth=0.5)
        axs[0][fig_num].set_xticks([])
        axs[0][fig_num].set_yticks([])
        axs[0][fig_num].set_xticks(np.arange(-0.5, t_in.shape[1], 1), minor=True)
        axs[0][fig_num].set_yticks(np.arange(-0.5, t_in.shape[0], 1), minor=True)
        axs[0][fig_num].grid(which='minor', color='lightgray', linestyle='-', linewidth=0.5)
        
        # Plot output
        '''
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f'Test Output', fontsize=16, fontweight='bold')  # Increase font size and set bold
        axs[1][fig_num].grid(True, color='gray', linestyle='-', linewidth=0.5)
        axs[1][fig_num].set_xticks([])
        axs[1][fig_num].set_yticks([])
        axs[1][fig_num].set_xticks(np.arange(-0.5, t_out.shape[1], 1), minor=True)
        axs[1][fig_num].set_yticks(np.arange(-0.5, t_out.shape[0], 1), minor=True)
        axs[1][fig_num].grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        '''
        # Skip plotting the test output
        axs[1][fig_num].axis('off')  # Turn off the axis for the test output grid


        fig_num += 1
    
    plt.tight_layout()

    # Save the figure if the save flag is set to True
    if save:
        fig.set_size_inches(figsize)  # Set figure size
        plt.savefig(filename, dpi=dpi, format=format, bbox_inches='tight')
        print(f"Plot saved as {filename} with dpi={dpi}, format={format}, and figsize={figsize}.")
    
    plt.show()

# Call the plot function with save=True to save the plot
plot_task(task, task_id, save=True, dpi=300, format="png", figsize=(12, 8))