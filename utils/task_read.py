import pandas as pd
import numpy as np

import os
import json

# Set the path to the ARC dataset
base_path = '/Users/vitsiozo/Desktop/MSc AI/Modules/Project/ARC/datasets/arc-prize-2024/'
# Load JSON data
def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

# Read the files
training_challenges = load_json(base_path + 'arc-agi_training_challenges.json')
training_solutions = load_json(base_path + 'arc-agi_training_solutions.json')

evaluation_challenges = load_json(base_path + 'arc-agi_evaluation_challenges.json')
evaluation_solutions = load_json(base_path + 'arc-agi_evaluation_solutions.json')

test_challenges = load_json(base_path + 'arc-agi_test_challenges.json')

# Do some data exploration

# Print the number of challenges and solutions
print (f'Number of training challenges = {len(training_challenges)}')
#print (f'Number of training solutions = {len(training_solutions)}')
print (f'Number of evaluation challenges = {len(evaluation_challenges)}')
#print (f'Number of evaluation solutions = {len(evaluation_solutions)}')
#print (f'Number of test challenges = {len(test_challenges)}')

# Print the names of the first 5 training challenges
'''
for i in range(5):
    t=list(training_challenges)[i]
    task = training_challenges[t]
    print(f'Task #{i}, {t}')
'''

# Ask user which task to load
while True:
    task_id = input('Enter the ID of the training task you want to plot: ')
    # Check if the file exists
    if training_challenges.get(task_id):
        break
    else:
        print('File not found. Please enter a valid task ID.')


task = training_challenges[task_id]
solution = training_solutions[task_id]



n_train_pairs = len(task['train'])
n_test_pairs = len(task['test'])

print(f'task {task_id}.json contains {n_train_pairs} training pairs and {n_test_pairs} test pairs')

# Function to format a 2D list (matrix-like formatting)
def print_matrix(matrix):
    for row in matrix:
        print(' '.join(map(str, row)))  # Join each element of the row with a space and print it

for i in range(n_train_pairs):
    print(f'\nTraining pair {i}')
    #print('Input:', task['train'][i]['input'])
    print_matrix(task['train'][i]['input'])  # Display input matrix

    print(f'Training pair {i} (Output):')
    #print('Output', task['train'][i]['output'])
    print_matrix(task['train'][i]['output'])  # Display output matrix

for j in range(n_test_pairs):
    print(f'\nTest pair {j}')
    #print('Input:', task['test'][j]['input'])
    print_matrix(task['test'][j]['input'])  # Display test input matrix

    print(f'Test pair {j} (Output):')
    #print('Output', solution[j])
    print_matrix(solution[j])  # Display test output matrix (from the solution)
