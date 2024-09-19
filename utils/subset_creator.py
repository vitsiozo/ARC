# Generate subset of tasks from user input

import os
import json
import numpy as np
import pandas as pd
import pprint

# Dictionary that holds the locations of task sets
task_sets = {
    'training' : {
        'challenges' : '/Users/vitsiozo/Desktop/MSc AI/Modules/Project/ARC/datasets/arc-prize-2024/arc-agi_training_challenges.json',
        'solutions' : '/Users/vitsiozo/Desktop/MSc AI/Modules/Project/ARC/datasets/arc-prize-2024/arc-agi_training_solutions.json',
    },
    'evaluation' : {
        'challenges' : '/Users/vitsiozo/Desktop/MSc AI/Modules/Project/ARC/datasets/arc-prize-2024/arc-agi_evaluation_challenges.json',
        'solutions' : '/Users/vitsiozo/Desktop/MSc AI/Modules/Project/ARC/datasets/arc-prize-2024/arc-agi_evaluation_solutions.json',
    }
}

def load_tasks_from_file(task_set):
    
    with open(task_set['challenges'], "r") as tasks:
        challenges = json.load(tasks)

    with open(task_set['solutions'], "r") as tasks:
        solutions = json.load(tasks)

    return challenges, solutions

# Load up training tasks
challenges, solutions = load_tasks_from_file(task_set=task_sets['training'])

# Function to allow the user to select tasks
def select_tasks(challenges, solutions, num_tasks=50):
    selected_challenges = {}
    selected_solutions = {}
    valid_keys = set(challenges.keys())
    
    print(f"Please select {num_tasks} task keys from the training set:")

    while len(selected_challenges) < num_tasks:
        # Ask the user to input task keys, either one by one or in bulk
        user_input = input(f"Enter task key(s) separated by commas (you need {num_tasks - len(selected_challenges)} more): ")
        entered_keys = [key.strip() for key in user_input.split(",")]  # Clean up the input

        for key in entered_keys:
            if key in valid_keys:
                if key not in selected_challenges:
                    selected_challenges[key] = challenges[key]
                    selected_solutions[key] = solutions[key]
                    print(f"Task {key} and its solution added.")
                else:
                    print(f"Task {key} is already selected.")
            else:
                print(f"Invalid key: {key}. Please enter a valid key.")

        print(f"Selected tasks so far: {list(selected_challenges.keys())}")

    return selected_challenges, selected_solutions

# Function to save selected data to JSON files
def save_selected_data_to_json(selected_challenges, selected_solutions, challenges_file='50_challenges.json', solutions_file='50_solutions.json'):
    # Save selected challenges to a JSON file
    with open(challenges_file, 'w') as challenges_json:
        json.dump(selected_challenges, challenges_json)  
        print(f"Selected challenges have been saved to {challenges_file}")

    # Save selected solutions to a JSON file
    with open(solutions_file, 'w') as solutions_json:
        json.dump(selected_solutions, solutions_json)
        print(f"Selected solutions have been saved to {solutions_file}")

# Use the function to select tasks
selected_challenges, selected_solutions = select_tasks(challenges, solutions)

# Use the function to save selected tasks and solutions to JSON files
save_selected_data_to_json(selected_challenges, selected_solutions)

print("New files created. Check the directory for the new files.")