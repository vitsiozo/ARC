# main.py

import os
import re
import json
import logging
from datetime import datetime
from typing import List, Tuple
from langchain_openai import ChatOpenAI  # To work with OpenAI
from langchain_anthropic import ChatAnthropic # To work with Anthropic
from langchain.prompts import PromptTemplate  # To help create our prompt
from abstraction import generate_abstracted_task, get_abstraction_method  # Import from abstraction.py

# Get API key for ChatGPT
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# Directory where tasks are stored
task_sets = {
    'training': {
        'challenges': '/Users/vitsiozo/Desktop/MSc AI/Modules/Project/ARC/datasets/50_challenges.json',  # Update this path
        'solutions': '/Users/vitsiozo/Desktop/MSc AI/Modules/Project/ARC/datasets/50_solutions.json',    # Update this path
    }
}

# Function to load ARC tasks from JSON files
def load_tasks_from_file(task_set):
    with open(task_set['challenges'], "r") as tasks:
        challenges = json.load(tasks)

    with open(task_set['solutions'], "r") as tasks:
        solutions = json.load(tasks)

    return challenges, solutions

def get_task_prediction(task_data, task_id, solutions, encoding, logger, test_input_index=0) -> List[List]:
    # Generate the abstracted task string by calling the function from abstraction.py
    task_string = generate_abstracted_task(task_data, task_id, encoding=encoding, test_input_index=test_input_index)

    # Get the abstraction method for this task
    abstraction_method_name = get_abstraction_method(task_id)
    logger.info(f"Using abstraction method '{abstraction_method_name}' for Task ID: {task_id}")
    multicolor_abs = ['na', 'mcccg', 'mcccg_d']  # List of multicolor abstraction methods

    # Prompt template
    prompt = PromptTemplate(
        template="You are a chatbot with human-like reasoning and abstraction capabilities.\n"
                 "We will engage in tasks that require reasoning and logic.\n"
                 "For each task, you will receive a few examples that demonstrate the transformation from an input to an output image.\n"
                 "After the examples you'll receive a new input image called test_input.\n"                
                 "Your task is to determine the corresponding output image from the transformation you can infer from the examples.\n"
                 "Use the same format as the one provided in the examples for your answer.\n"
                 "Do not give any justification or extra text for your answer, just provide the output image.\n{task_string}\n\nYour Response:",
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
        multicolor_abs=multicolor_abs,
        logger=logger
    )

    if not grid:
        logger.error(f"Failed to reconstruct grid for Task {task_id}, Test Input {test_input_index}.")
        grid = []  # Assign empty grid

    # Format the reconstructed grid for readability
    formatted_grid = "[\n" + ",\n".join(str(row) for row in grid) + "\n]"

    # Log the reconstructed grid
    logger.info(f"Reconstructed grid for Task {task_id}, Test Input {test_input_index}:\n{formatted_grid}")

    # Get the correct solution
    correct_solution = solutions[task_id][test_input_index]

    # Format the correct solution for readability
    formatted_solution = "[\n" + ",\n".join(str(row) for row in correct_solution) + "\n]"

    # Log the correct solution for comparison
    logger.info(f"Correct solution for Task {task_id}, Test Input {test_input_index}:\n{formatted_solution}")

    return grid

def run_model(challenges, solutions, logger, NUM_ATTEMPTS=1, RETRY_ATTEMPTS=3, NUM_TASKS=None, encoding='object_json'):
    results = {}
    task_ids = list(challenges.keys())  # List of task ids

    if NUM_TASKS:
        task_ids = task_ids[:NUM_TASKS]  # Limit the number of tasks if specified

    for i, task_id in enumerate(task_ids):
        task_attempts = []  # List to store all attempts for the current task

        task_data = challenges[task_id]
        
        # Go through each test pair (input/output pair) in the task
        for t in range(len(task_data['test'])):
            logger.info(f"Processing Task ID: {task_id}, Test Pair #{t + 1}")

            pair_attempts = {}  # Dictionary to store attempts for the current test pair

            # Perform prediction attempts for each test input
            for attempt in range(1, NUM_ATTEMPTS + 1):
                attempt_key = f"attempt_{attempt}"
                pair_attempts[attempt_key] = []  # Initialize empty prediction for this attempt

                # Retry mechanism in case of failure
                for retry in range(RETRY_ATTEMPTS):
                    try:
                        logger.info(f"Attempting prediction #{attempt}, retry #{retry + 1}")

                        # Get the prediction for the task and specific test input
                        prediction = get_task_prediction(task_data, task_id, solutions, encoding=encoding, logger=logger, test_input_index=t)

                        # Store the valid prediction
                        pair_attempts[attempt_key] = prediction
                        break  # Exit the retry loop after a successful prediction
                    except Exception as e:
                        if logger:  # Ensure logger is not None
                             logger.warning(f"Prediction failed for attempt #{attempt}, retry #{retry + 1}: {e}")
                        else:
                             print(f"Prediction failed for attempt #{attempt}, retry #{retry + 1}: {e}")
                        if retry == RETRY_ATTEMPTS - 1:
                            pair_attempts[attempt_key] = []

            # Append all attempts for this test pair to task_attempts
            task_attempts.append(pair_attempts)

        # Store the results for this task
        results[task_id] = task_attempts

        # Stop if the number of tasks has been limited
        if NUM_TASKS is not None and i + 1 == NUM_TASKS:
            break

    return results

def parse_model_response_to_grid(response_text, encoding='object_json', mode='Output', abstraction='nbccg', multicolor_abs=[], logger=None):
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

    if logger is None:
        logger = logging.getLogger(__name__)

    # Remove all whitespace and newlines for easier parsing
    logger.debug(f"Raw response text: {response_text}")
    response_text = response_text.replace(' ', '').replace('\n', '')

    # Dictionary to map words to digits for color encoding
    word_to_digit = {
        'black': 0, 'blue': 1, 'red': 2, 'green': 3,
        'yellow': 4, 'gray': 5, 'purple': 6, 'orange': 7,
        'cyan': 8, 'brown': 9, "0": 0, "1": 1, "2": 2,
        "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9
    }

    if encoding == "object_descriptor":
        # Remove spaces in the response
        response_text = response_text.replace(' ', '')

        # Extract the size of the image
        size_pattern = r'(?:Image|Input|Output)\s*size\s*:\s*\((\d+),\s*(\d+)\)'
        size_match = re.findall(size_pattern, response_text)
        if not size_match:
            logger.error("Failed to find image size in the model's response.")
            return "parsing_error"
        height, width = int(size_match[0][0]), int(size_match[0][1])
        grid = [[0 for _ in range(width)] for _ in range(height)]

        # Define the object pattern based on whether multicolor abstraction is used
        if abstraction in multicolor_abs:
            object_pattern = r'Object(\d+):coordinates=\[(.*?)\],color=\[(.*?)\],size=(\d+)'
        else:
            object_pattern = r'Object(\d+):coordinates=\[(.*?)\],color=(\d+),size=(\d+)'

        object_matches = re.findall(object_pattern, response_text)

        if not object_matches:
            logger.error("Failed to find objects in the model's response.")
            return "parsing_error"

        for match in object_matches:
            node_id, coordinates_str, color, size = match
            coordinates = [tuple(map(int, coord.split(','))) for coord in coordinates_str.strip(")(").split('),(')]

            if abstraction in multicolor_abs:
                color_list = [int(c) for c in color.split(',')]
                for indx, (rnum, cnum) in enumerate(coordinates):
                    try:
                        grid[rnum][cnum] = color_list[indx]
                    except IndexError:
                        pass
            else:
                color = int(color)
                for (rnum, cnum) in coordinates:
                    try:
                        grid[rnum][cnum] = color
                    except IndexError:
                        pass

    elif encoding == "object_json" or encoding == "object_json_words":
        response_text = response_text.replace(' ', '')

        # Initialize output grid
        size_pattern = r'(?:Image|Input|Output)\s*size\s*:\s*\((\d+),\s*(\d+)\)'
        size_match = re.findall(size_pattern, response_text)
        if not size_match:
            logger.error("Failed to find image size in the model's response.")
            return "parsing_error"
        height, width = int(size_match[0][0]), int(size_match[0][1])
        grid = [[0 for _ in range(width)] for _ in range(height)]

        # Find object lines in the input string
        if abstraction in multicolor_abs:
            object_pattern = r'"coordinates":\[(.*?)\],"color":\[(.*?)\],"size":(\d+)'
        else:
            if encoding != "object_json_words":
                object_pattern = r'"coordinates":\[(.*?)\],"color":(\d+),"size":(\d+)'
            else:
                object_pattern = r'"coordinates":\[(.*?)\],"color":(.*?),"size":(\d+)'

        object_matches = re.findall(object_pattern, response_text)

        if not object_matches:
            logger.error("Failed to find objects in the model's response.")
            return "parsing_error"

        for match in object_matches:
            coordinates_str, color, size = match
            coordinates = [tuple(map(int, coord.split(','))) for coord in coordinates_str.strip("][").split('],[')]

            if abstraction in multicolor_abs:
                color_list = [word_to_digit[c.strip("\"\'")] for c in color.split(',')]
                for indx, (rnum, cnum) in enumerate(coordinates):
                    try:
                        grid[rnum][cnum] = color_list[indx]
                    except IndexError:
                        pass
            else:
                color = word_to_digit[color.strip("\"\'")]
                for (rnum, cnum) in coordinates:
                    try:
                        grid[rnum][cnum] = color
                    except IndexError:
                        pass

    elif encoding == "object_json_w_edge":
        response_text = response_text.replace(' ', '')

        # Initialize output grid
        size_pattern = r'(?:Image|Input|Output)\s*size\s*:\s*\((\d+),\s*(\d+)\)'
        size_match = re.findall(size_pattern, response_text)
        if not size_match:
            logger.error("Failed to find image size in the model's response.")
            return "parsing_error"
        height, width = int(size_match[0][0]), int(size_match[0][1])
        grid = [[0 for _ in range(width)] for _ in range(height)]

        # Find object lines in the input string
        if abstraction in multicolor_abs:
            object_pattern = r'"coordinates":\[(.*?)\],"color":\[(.*?)\],"size":(\d+),"id":(\d+),"neighbors":\[(.*?)\]'
        else:
            object_pattern = r'"coordinates":\[(.*?)\],"color":(\d+),"size":(\d+),"id":(\d+),"neighbors":\[(.*?)\]'

        object_matches = re.findall(object_pattern, response_text)

        if not object_matches:
            logger.error("Failed to find objects in the model's response.")
            return "parsing_error"

        for match in object_matches:
            coordinates_str, color, size, id, neighbor = match
            coordinates = [tuple(map(int, coord.split(','))) for coord in coordinates_str.strip("][").split('],[')]

            if abstraction in multicolor_abs:
                color_list = [int(c) for c in color.split(',')]
                for indx, (rnum, cnum) in enumerate(coordinates):
                    try:
                        grid[rnum][cnum] = color_list[indx]
                    except IndexError:
                        pass
            else:
                color = int(color)
                for (rnum, cnum) in coordinates:
                    try:
                        grid[rnum][cnum] = color
                    except IndexError:
                        pass

    elif encoding == "object_descriptor_w_edge":
        response_text = response_text.replace(' ', '')

        # Initialize output grid
        size_pattern = r'(?:Image|Input|Output)\s*size\s*:\s*\((\d+),\s*(\d+)\)'
        size_match = re.findall(size_pattern, response_text)
        if not size_match:
            logger.error("Failed to find image size in the model's response.")
            return "parsing_error"
        height, width = int(size_match[0][0]), int(size_match[0][1])
        grid = [[0 for _ in range(width)] for _ in range(height)]

        if abstraction in multicolor_abs:
            object_pattern = r'Object(\d+):coordinates=\[(.*?)\],color=\[(.*?)\],size=(\d+),neighbors=\[(.*?)\]'
        else:
            object_pattern = r'Object(\d+):coordinates=\[(.*?)\],color=(\d+),size=(\d+),neighbors=\[(.*?)\]'

        object_matches = re.findall(object_pattern, response_text)

        if not object_matches:
            logger.error("Failed to find objects in the model's response.")
            return "parsing_error"

        for match in object_matches:
            node_id, coordinates_str, color, size, neighbors = match
            coordinates = [tuple(map(int, coord.split(','))) for coord in coordinates_str.strip(")(").split('),(')]

            if abstraction in multicolor_abs:
                color_list = [int(c) for c in color.split(',')]
                for indx, (rnum, cnum) in enumerate(coordinates):
                    try:
                        grid[rnum][cnum] = color_list[indx]
                    except IndexError:
                        pass
            else:
                color = int(color)
                for (rnum, cnum) in coordinates:
                    try:
                        grid[rnum][cnum] = color
                    except IndexError:
                        pass

    else:
        logger.error(f"Encoding '{encoding}' not recognized.")
        return "parsing_error"

    return grid


def score_results(results, solutions, logger) -> Tuple[float, int]:
    total_score = 0
    total_tasks = 0  # Count total tasks, each task can have multiple test inputs

    # Iterate over tasks
    for task_id, task_predictions in results.items():
        task_solutions = solutions[task_id]  # List of correct solutions for each test input
        num_test_inputs = len(task_solutions)  # Number of test inputs for this task
        total_tasks += 1  # Count total tasks

        task_score = 0  # Track score for this task

        # Calculate weight per test input (e.g., 0.5 for tasks with 2 test inputs)
        test_weight = 1 / num_test_inputs

        # Iterate over test pairs in the task
        for test_pair_index, test_pair_attempts in enumerate(task_predictions):
            logger.info(f"Scoring Task {task_id}, Test Input #{test_pair_index + 1}")
            pair_correct = False

            # Iterate over multiple attempts for this test pair (if applicable)
            for attempt_key, attempt in test_pair_attempts.items():
                if attempt == task_solutions[test_pair_index]:
                    logger.info(f"Task ID {task_id}, Test Input {test_pair_index + 1}, {attempt_key}: Correct")
                    pair_correct = True
                    break
                else:
                    logger.debug(f"Task ID {task_id}, Test Input {test_pair_index + 1}, {attempt_key}: Incorrect")

            if pair_correct:
                task_score += test_weight  # Add the weighted score for this test input

        total_score += task_score  # Add the task's score to the total score

    # Log the final score
    logger.info(f"Total score: {total_score}, Total tasks scored: {total_tasks}")
    return {
        'total_score': total_score,
        'total_tasks_scored': total_tasks
    }

def main(task_set='training'):
    global model_name, llm

    # Prompt the user to select the model
    print("Select the model to use:")
    print("1: GPT-3.5 (gpt-3.5-turbo)")
    print("2: GPT-4o Mini (gpt-4o-mini)")
    print("3: GPT-4o (gpt-4o)")
    print("4: o1 Mini (o1-mini)")
    print("5: o1 Preview (o1-preview)")
    print("6: Claude-3.5 Sonnet (claude-3-5-sonnet-20240620)")
    
    # Ask user to select the model
    while True:
        model_choice = input("Enter the number corresponding to the model (1 to 6): ")
        if model_choice == "1":
            model_name = "gpt-3.5-turbo"
            llm = ChatOpenAI(model=model_name, api_key=OPENAI_API_KEY, max_tokens=3000, temperature=0.0)
            break
        elif model_choice == "2":
            model_name = "gpt-4o-mini"
            llm = ChatOpenAI(model=model_name, api_key=OPENAI_API_KEY, max_tokens=3000, temperature=0.0)
            break
        elif model_choice == "3":
            model_name = "gpt-4o"
            llm = ChatOpenAI(model=model_name, api_key=OPENAI_API_KEY, max_tokens=3000, temperature=0.0)
            break
        elif model_choice == "4":
            model_name = "o1-mini"
            llm = ChatOpenAI(model=model_name, api_key=OPENAI_API_KEY, temperature=1.0)
            break
        elif model_choice == "5":
            model_name = "o1-preview"
            llm = ChatOpenAI(model=model_name, api_key=OPENAI_API_KEY, temperature=1.0)
            break
        elif model_choice == "6":
            model_name = "claude-3-5-sonnet-20240620"
            llm = ChatAnthropic(model=model_name, api_key=ANTHROPIC_API_KEY, max_tokens=3000, temperature=0.0)
            break
        else:
            print("Invalid input. Please enter 1, 2, 3, 4, 5, or 6.")

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

    # Directory for logging files
    log_dir = '/Users/vitsiozo/Desktop/MSc AI/Modules/Project/ARC/log_obj_output' 

    # Generate log file name based on the model and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = os.path.join(log_dir, f"{model_name}_{encoding}_{timestamp}.log")

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

    # Load datasets
    challenges, solutions = load_tasks_from_file(task_set=task_sets[task_set])

    # Run the model
    test_results = run_model(challenges, solutions, logger, NUM_TASKS=NUM_TASKS, encoding=encoding)

    # Score the results
    score_result = score_results(logger=logger, results=test_results, solutions=solutions)

    logger.info(f"Model name: {model_name}, Model temperature: {llm.temperature}")
    logger.info(f"Final score: {score_result['total_score']} of {score_result['total_tasks_scored']} "
                f"({round(score_result['total_score'] / score_result['total_tasks_scored'] * 100, 2)}%)")

# Start the program
if __name__ == "__main__":
    main(task_set='training')
