# main.py

import os
import re
import json
import logging
from datetime import datetime
from typing import List, Tuple
from langchain_openai import ChatOpenAI  # To work with OpenAI
from langchain.prompts import PromptTemplate  # To help create our prompt
from abstraction import generate_abstracted_task, get_abstraction_method  # Import from abstraction.py

# Get API key for ChatGPT
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Choose the model to use
model_name = 'gpt-4o-mini'  # Update this to the model you have access to
llm = ChatOpenAI(model_name=model_name, openai_api_key=OPENAI_API_KEY, max_tokens=3000, temperature=0.0)

# Directory for logging files
log_dir = '/Users/vitsiozo/Desktop/MSc AI/Modules/Project/ARC/log_obj_output'  # Update this path

# Generate log file name based on the model and timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_name = os.path.join(log_dir, f"{model_name}_{timestamp}.log")

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Set log level
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_name),  # Log to file
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger(__name__)

task_sets = {
    'training': {
        'challenges': '/Users/vitsiozo/Desktop/MSc AI/Modules/Project/ARC/datasets/50_challenges.json',  # Update this path
        'solutions': '/Users/vitsiozo/Desktop/MSc AI/Modules/Project/ARC/datasets/50_solutions.json',    # Update this path
    }
}

def load_tasks_from_file(task_set):
    with open(task_set['challenges'], "r") as tasks:
        challenges = json.load(tasks)

    with open(task_set['solutions'], "r") as tasks:
        solutions = json.load(tasks)

    return challenges, solutions

def get_task_prediction(task_data, task_id, encoding) -> List[List]:
    # Generate the abstracted task string by calling the function from abstraction.py
    task_string = generate_abstracted_task(task_data, task_id, encoding=encoding)

    # Get the abstraction method for this task
    abstraction_method_name = get_abstraction_method(task_id)
    logger.info(f"Using abstraction method '{abstraction_method_name}' for Task ID: {task_id}")
    multicolor_abs = ['mcccg', 'mcccg_d']  # List of multicolor abstraction methods
    mode = 'Output'  # Assuming 'Output' is used in the response

    # Prompt template
    prompt = PromptTemplate(
        template="You are a reasoning assistant adept at solving abstract reasoning tasks. "
                 "Given the abstracted representations of training examples (inputs and outputs), "
                 "infer the transformation and apply it to the test input to produce the output grid. "
                 "Provide the output grid exactly in the same format as the examples, including 'Image size' and 'Objects'. "
                 "Ensure the 'Image size' is in the format (height, width). "
                 "Do not include any explanations or extra text.\n{task_string}\n\nYour Response:",
        input_variables=["task_string"]
    )

    # Generate the full prompt
    formatted_prompt = prompt.format(task_string=task_string)

    # Log the prompt
    logger.info(f"Prompt:\n{formatted_prompt}")

    # Call the model and get the prediction
    response = llm.invoke(formatted_prompt)

    # Extract the actual prediction from the response
    prediction_string = response.content.strip()

    # Log the raw response
    logger.info(f"Raw model response:\n{prediction_string}")

    # Parse the model's response to reconstruct the grid
    grid = parse_model_response_to_grid(
        prediction_string,
        encoding=encoding,
        abstraction=abstraction_method_name,
        multicolor_abs=multicolor_abs
    )

    if not grid:
        logger.error("Failed to reconstruct grid from the model's response.")
        grid = []  # Assign empty grid

    # Log the reconstructed grid
    logger.info(f"Reconstructed grid for Task ID {task_id}:\n{grid}")

    return grid

def run_model(challenges, solutions, NUM_TASKS=None, encoding='object_json'):
    results = {}

    task_ids = list(challenges.keys())
    if NUM_TASKS:
        task_ids = task_ids[:NUM_TASKS]

    for task_id in task_ids:
        logger.info(f"Processing Task ID: {task_id}")
        task_data = challenges[task_id]

        # Get the prediction for the task
        prediction = get_task_prediction(task_data, task_id, encoding=encoding)

        # Store the prediction
        results[task_id] = prediction

    return results

def parse_model_response_to_grid(response_text, encoding='object_json', mode='Output', abstraction='nbccg', multicolor_abs=[]):
    """
    Parses the model's response and reconstructs the grid.

    Args:
        response_text (str): The raw response from the model.
        encoding (str): The encoding used in the model's response.
        mode (str): Either 'Input' or 'Output', depending on the section.
        abstraction (str): The abstraction method used.
        multicolor_abs (list): List of abstraction methods that use multicolor.

    Returns:
        list: A 2D list representing the grid.
    """
    # Remove all whitespace and newlines for easier parsing
    response_text = response_text.replace(' ', '').replace('\n', '')

    word_to_digit = {
        'black': 0, 'blue': 1, 'red': 2, 'green': 3,
        'yellow': 4, 'gray': 5, 'purple': 6, 'orange': 7,
        'cyan': 8, 'brown': 9, "0": 0, "1": 1, "2": 2,
        "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9
    }

    # Parsing size pattern for grid dimensions
    size_pattern = rf'{mode}size:\((\d+),(\d+)\)'
    size_match = re.findall(size_pattern, response_text)
    if not size_match:
        logger.error("Failed to find image size in the model's response.")
        return []
    
    height, width = int(size_match[0][0]), int(size_match[0][1])
    grid = [[0 for _ in range(width)] for _ in range(height)]

    # Handle object descriptors (including with edges)
    if encoding in ["object_descriptor", "object_descriptor_w_edge"]:
        if abstraction in multicolor_abs:
            object_pattern = r'Object(\d+):coordinates=\[(.*?)\],color=\[(.*?)\],size=(\d+)'
            if encoding == "object_descriptor_w_edge":
                object_pattern += r',neighbors=\[(.*?)\]'
        else:
            object_pattern = r'Object(\d+):coordinates=\[(.*?)\],color=(\d+),size=(\d+)'
            if encoding == "object_descriptor_w_edge":
                object_pattern += r',neighbors=\[(.*?)\]'
        
        object_matches = re.findall(object_pattern, response_text)
        if not object_matches:
            logger.error("Failed to find objects in the model's response.")
            return []

        for match in object_matches:
            node_id, coordinates_str, color, size = match[:4]  # neighbors are optional
            coordinates = [tuple(map(int, coord.split(','))) for coord in coordinates_str.strip(")(").split('),(')]

            # Handle multicolor abstraction
            if abstraction in multicolor_abs:
                color = [int(c) for c in color.split(',')]
                for indx, (rnum, cnum) in enumerate(coordinates):
                    try:
                        grid[rnum][cnum] = color[indx]
                    except IndexError:
                        pass
            else:
                color = int(color)
                for (rnum, cnum) in coordinates:
                    try:
                        grid[rnum][cnum] = color
                    except IndexError:
                        pass
        return grid

    # Handle object_json or object_json_words encoding
    elif encoding in ["object_json", "object_json_words", "object_json_w_edge"]:
        if abstraction in multicolor_abs:
            object_pattern = r'"coordinates":\[(.*?)\],"color":\[(.*?)\],"size":(\d+)'
        else:
            if encoding == "object_json_words":
                object_pattern = r'"coordinates":\[(.*?)\],"color":"(.*?)","size":(\d+)'
            else:
                object_pattern = r'"coordinates":\[(.*?)\],"color":(\d+),"size":(\d+)'

        object_matches = re.findall(object_pattern, response_text)
        if not object_matches:
            logger.error("Failed to find objects in the model's response.")
            return []

        for match in object_matches:
            coordinates_str, color, size = match
            coordinates = [tuple(map(int, coord.split(','))) for coord in coordinates_str.strip("][").split('],[')]

            # Handle multicolor abstraction
            if abstraction in multicolor_abs:
                color = [int(c) for c in color.split(',')]
                for indx, (rnum, cnum) in enumerate(coordinates):
                    try:
                        grid[rnum][cnum] = color[indx]
                    except IndexError:
                        pass
            else:
                if encoding == "object_json_words":
                    color = word_to_digit.get(color.strip("\"\'"), 0)
                else:
                    color = int(color)
                for (rnum, cnum) in coordinates:
                    try:
                        grid[rnum][cnum] = color
                    except IndexError:
                        pass
        return grid

    # Handle graph-like encodings
    elif encoding == "node_edge_set":
        # Parse nodes and edges (for `node_edge_set` encoding)
        node_pattern = r'Node(\d+):coordinates=\[(.*?)\],color=(\d+),size=(\d+)'
        node_matches = re.findall(node_pattern, response_text)
        if not node_matches:
            logger.error("Failed to find nodes in the model's response.")
            return []

        for match in node_matches:
            node_id, coordinates_str, color, size = match
            coordinates = [tuple(map(int, coord.split(','))) for coord in coordinates_str.strip(")(").split('),(')]
            color = int(color)
            for (rnum, cnum) in coordinates:
                try:
                    grid[rnum][cnum] = color
                except IndexError:
                    pass
        return grid

    else:
        logger.error(f"Encoding '{encoding}' not supported.")
        return []

def score_results(results, solutions) -> Tuple[float, int]:
    total_score = 0
    total_tasks = len(results)

    for task_id, prediction in results.items():
        correct_solution = solutions[task_id][0]  # Assuming first test input
        if prediction == correct_solution:
            total_score += 1
            logger.info(f"Task ID {task_id}: Correct")
        else:
            logger.info(f"Task ID {task_id}: Incorrect")
            # Optionally log the expected and predicted solutions
            logger.debug(f"Expected solution:\n{correct_solution}")
            logger.debug(f"Predicted solution:\n{prediction}")

    logger.info(f"Total score: {total_score}/{total_tasks}")
    return {
        'total_score': total_score,
        'total_tasks_scored': total_tasks
    }

def main(task_set='training'):
    # Load datasets
    challenges, solutions = load_tasks_from_file(task_set=task_sets[task_set])

    # Prompt the user to select the encoding type
    print("Please select the encoding type:")
    print("1) Object JSON")
    print("2) Object Descriptor")
    print("3) Object JSON with edge")
    print("4) Object Descriptor with edge")
    print("5) Object JSON with words")

    while True:
        try:
            encoding_choice = int(input("Select 1-5: "))
            if encoding_choice not in range(1, 6):
                raise ValueError("Invalid selection. Please choose a number between 1 and 5.")
            break
        except ValueError as e:
            print(e)

    # Map user choice to the corresponding encoding type
    encoding_map = {
        1: 'object_json',
        2: 'object_descriptor',
        3: 'object_json_w_edge',
        4: 'object_descriptor_w_edge',
        5: 'object_json_words'
    }

    # Get the selected encoding type
    encoding = encoding_map[encoding_choice]

    # Ask the user for the number of tasks they want to run
    while True:
        try:
            NUM_TASKS = input("Enter the number of tasks you want to process (or 'all' to process all tasks): ")
            if NUM_TASKS.lower() == 'all':
                NUM_TASKS = None  # None will process all tasks
                break
            else:
                NUM_TASKS = int(NUM_TASKS)  # Convert input to integer
                if NUM_TASKS < 1:
                    logger.warning("Please enter a positive number.")
                else:
                    break  # Break the loop if input is valid
        except ValueError:
            logger.error("Invalid input. Please enter a numerical value or 'all'.")

    # Run the model
    test_results = run_model(challenges, solutions, NUM_TASKS=NUM_TASKS, encoding=encoding)

    # Score the results
    score_result = score_results(results=test_results, solutions=solutions)

    logger.info(f"Model name: {model_name}, Model temperature: {llm.temperature}")
    logger.info(f"Final score: {score_result['total_score']} of {score_result['total_tasks_scored']} "
                f"({round(score_result['total_score'] / score_result['total_tasks_scored'] * 100, 2)}%)")

# Start the program
if __name__ == "__main__":
    main(task_set='training')
