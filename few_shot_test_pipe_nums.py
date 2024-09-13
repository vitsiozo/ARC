import os
import json
import re
import logging
from datetime import datetime
from typing import List, Tuple
from langchain_openai import ChatOpenAI  # To work with OpenAI
from langchain_core.prompts import PromptTemplate  # To help create our prompt

# Get api key for chatgpt
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Choose the model to use
model_name = 'gpt-4o-mini'
llm = ChatOpenAI(model=model_name, api_key=OPENAI_API_KEY, max_tokens=3000, temperature=0.4)

# Directory for logging files
log_dir = '/Users/vitsiozo/Desktop/MSc AI/Modules/Project/ARC/log_output'

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
        'challenges': '/Users/vitsiozo/Desktop/MSc AI/Modules/Project/ARC/50_challenges.json',
        'solutions': '/Users/vitsiozo/Desktop/MSc AI/Modules/Project/ARC/50_solutions.json',
    }
}

def load_tasks_from_file(task_set):
    with open(task_set['challenges'], "r") as tasks:
        challenges = json.load(tasks)

    with open(task_set['solutions'], "r") as tasks:
        solutions = json.load(tasks)

    return challenges, solutions

def json_task_to_string(challenge_tasks: dict, task_id: str, test_input_index: int) -> str:
    json_task = challenge_tasks[task_id]

    final_output = "Training Examples\n"

    train_tasks = json_task['train']
    test_task = json_task['test']

    for i, task in enumerate(train_tasks):
        final_output += f"Example {i + 1}: Input\n"

        # Convert input grid to the new format with pipes
        for row in task['input']:
            row_str = '|'.join(str(num) for num in row)
            final_output += f"{row_str}\n"

        final_output += f"\nExample {i + 1}: Output\n"

        # Convert output grid to the new format with pipes
        for row in task['output']:
            row_str = '|'.join(str(num) for num in row)
            final_output += f"{row_str}\n"

        final_output += "\n"

    final_output += "Test Input\n"

    # Convert test input grid to the new format with pipes
    for row in test_task[test_input_index]['input']:
        row_str = '|'.join(str(num) for num in row)
        final_output += f"{row_str}\n"

    final_output += "\nYour Response:"

    return final_output


def parse_prediction(prediction_string: str) -> List[List[int]]:
    # Split the string into lines
    lines = prediction_string.strip().split('\n')
    # For each line, split by '|', and convert to integers
    prediction = []
    for line in lines:
        if line.strip() == '':
            continue  # Skip empty lines
        row_strs = line.strip().split('|')
        row = [int(num) for num in row_strs]
        prediction.append(row)
    return prediction

def get_task_prediction(challenge_tasks, solutions, task_id, test_input_index) -> List[List]:

    # Get the string representation of the task
    task_string = json_task_to_string(challenge_tasks, task_id, test_input_index)

    # Prompt template 1
    '''prompt = PromptTemplate(
        template="You are a chatbot with human-like reasoning and inference capabilities, adept at solving tasks concisely. "
                 "Let's engage in reasoning and logic-based tasks. Each task will demonstrate a transformation from an input to an output grid. "
                 "At the end, you'll receive a new input grid. "
                 "Your task is to determine its corresponding output based on the logic of transformations that is found in the examples. "
                 "Do not give any justification for your answer, just provide a list of lists as the output.\n\n{task_string}\n",
        input_variables=["task_string"]
    )'''

    # Prompt template 2
    prompt = PromptTemplate(
    template=(
            "You are a chatbot that is adept at finding patterns and solving reasoning tasks. "
             "Let's engage in a series of puzzles where you are asked to find the pattern in a set of examples and based on that to make a prediction on a new input. "
             "Visualize the set of numbers presented as a 2-dimensional grid. "
             "Each row of numbers represents a row of pixels in the grid. "
             "Each number on this grid represents a different color. Number 0 is black and represents the background. "
             "The arrangement of the other numbers signifies different objects, shapes, or patterns present on the grid. "
             "At the beginning of each task, you will be presented with a set of examples. "
             "You will see the example input followed by the example output. "
             "To get from the example input to the example output, a specific pattern or transformation has been applied. "
             "The data will be presented as numbers separated by '|', and newlines to designate new rows. "
             "Your task is to identify this pattern or transformation and apply it to the test input to get the final output. "
             "Do not give any justification for your answer; just provide the output grid in the same format: no brackets or commas, just numbers, with each line representing a row."
             "\n\n{task_string}\n"
    ),
    input_variables=["task_string"]
)

    # Generate the full prompt
    formatted_prompt = prompt.format(task_string=task_string)

    # Log the prompt
    logger.info(f"Prompt:\n{formatted_prompt}")

    # Call the model and get the prediction
    response = llm.invoke(formatted_prompt)

    # Extract the actual prediction from the response content
    prediction_string = response.content

    # Parse the prediction string into a list of lists of integers
    try:
        prediction = parse_prediction(prediction_string)
    except Exception as e:
        logger.error(f"Failed to parse prediction: {e}")
        prediction = []  # Assign an empty list if parsing fails

    # Log the prediction
    logger.info(f"Prediction:\n{prediction_string}\n")

    # Also log the correct solution
    correct_solution = solutions[task_id][test_input_index]  # Get the correct solution
     # Convert the correct solution to the new format for logging
    solution_lines = ['|'.join(map(str, row)) for row in correct_solution]
    solution_string = '\n'.join(solution_lines)
    logger.info(f"Solution:\n{solution_string}\n")

    return prediction

def run_model(challenges, solutions, NUM_ATTEMPTS=2, RETRY_ATTEMPTS=3, NUM_TASKS=None):

    # A dict to hold the results returned after all predictions are made
    results = {}

    # Run through each task  
    for i, task_id in enumerate(challenges):
        task_attempts = [] # List to store all attempts for the current task

        # Go through each test pair to get a prediction. 96% of challenges have just 1 pair.
        for t, pair in enumerate(challenges[task_id]['test']):
            logger.info(f"Starting task #{i + 1} ({task_id}), pair #{t+1}")

            # Dictionary to store attempts for the current test pair
            pair_attempts = {}

            # Run through each prediction attempt
            for attempt in range(1, NUM_ATTEMPTS + 1):
                attempt_key = f"attempt_{attempt}"
                pair_attempts[attempt_key] = [] # Init your attempt

                # Try to get a prediction, with retries in case of failure
                for retry in range(RETRY_ATTEMPTS):
                    try:
                        logger.info(f"    Predicting attempt #{attempt}, retry #{retry + 1}")
                        prediction = get_task_prediction(challenge_tasks=challenges,
                                                         solutions=solutions,
                                                         task_id=task_id,
                                                         test_input_index=t)

                        # If you get a valid prediction (list of lists of ints) with no error, then log the attempt
                        pair_attempts[attempt_key] = prediction
                        break  # Break the retry loop if prediction is successful
                    except Exception as e:
                        logger.warning(f"Retrying: {e}")
                        if retry == RETRY_ATTEMPTS - 1:
                            pair_attempts[attempt_key] = [] # Assign None if all retries fail

            # After you get your attempts, append them to the task attempts
            task_attempts.append(pair_attempts)

        # Append the task attempts to the submission with the task_id as the key
        results[task_id] = task_attempts
        #print(f"Result for task {task_id}:\n{task_attempts}")

        # If you want to stop after N tasks, uncomment the below
        if NUM_TASKS is not None and i + 1 == NUM_TASKS:
            break

    return results

def score_results(results, solutions) -> Tuple[float, int]:
    total_score = 0
    total_tasks = 0

    # Loop through each task in your results to grade it
    for task_id, task_attempts in results.items():
        total_tasks += 1
        task_score = 0
        num_pairs = len(task_attempts)

        # Go through each task. Most will only have 1 pair.
        for pair_index, pair_attempts in enumerate(task_attempts):
            logger.info(f"Scoring Task {task_id} pair #{pair_index+1}")
            pair_correct = False

            # Look at both of your attempts
            for attempt_key, attempt in pair_attempts.items():

                # If the attempt matches the solution, then it's correct
                if attempt == solutions[task_id][pair_index]:
                    logger.info(f"Task Id {task_id} pair {pair_index+1} {attempt_key} matches solution")
                    pair_correct = True
                    break # If it is correct, log it and break the loop

            if pair_correct:
                task_score += 1

        task_score /= num_pairs
        total_score += task_score
        #print(f"Score for Task {task_id}: {task_score}")  # Debug: print each task score

    logger.info(f"Total score: {total_score}, Total tasks scored: {total_tasks}")
    return {
        'total_score': total_score,
        'total_tasks_scored': total_tasks
    }

def main(task_set='training', NUM_TASKS=None):
    # Load datasets
    challenges, solutions = load_tasks_from_file(task_set=task_sets[task_set])

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
    test_results = run_model(challenges, solutions, NUM_TASKS=NUM_TASKS)

    # Score the results
    score_result = score_results(results=test_results, solutions=solutions)

    logger.info(f"Model name: {model_name}, Model temperature: {llm.temperature}")
    logger.info(f"Final score: {score_result['total_score']} of {score_result['total_tasks_scored']} "
          f"({round(score_result['total_score'] / score_result['total_tasks_scored'] * 100, 2)}%)")

# Start the program
if __name__ == "__main__":
    main(task_set='training')
