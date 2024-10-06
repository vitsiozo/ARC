import os
import re
import json
import logging
from datetime import datetime
from typing import List, Tuple
from langchain_openai import ChatOpenAI  # To work with OpenAI
from langchain_anthropic import ChatAnthropic # To work with Anthropic
from langchain_core.prompts import PromptTemplate  # To help create our prompt

# Get api keys for OpenAI and Anthropic
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# Directory for logging files
log_dir = '/Users/vitsiozo/Desktop/MSc AI/Modules/Project/ARC/log_txt_output/cot_few_shot/words'

# Directory where tasks are stored
task_sets = {
    'training': {
        'challenges': '/Users/vitsiozo/Desktop/MSc AI/Modules/Project/ARC/datasets/50_challenges.json',
        'solutions': '/Users/vitsiozo/Desktop/MSc AI/Modules/Project/ARC/datasets/50_solutions.json',
    }
}

# Define the mapping between numbers and color words
number_to_color = {
    0: 'black',
    1: 'blue',
    2: 'red',
    3: 'green',
    4: 'yellow',
    5: 'gray',
    6: 'magenta',
    7: 'orange',
    8: 'cyan',
    9: 'brown'
}

# Create the reverse mapping from color words to numbers
color_to_number = {v: k for k, v in number_to_color.items()}

def load_tasks_from_file(task_set):
    with open(task_set['challenges'], "r") as tasks:
        challenges = json.load(tasks)

    with open(task_set['solutions'], "r") as tasks:
        solutions = json.load(tasks)

    return challenges, solutions

# Function to convert a task to a formatted string
def json_task_to_string(challenge_tasks: dict, task_id: str, test_input_index: int) -> str:
    json_task = challenge_tasks[task_id]

    final_output = "Training Examples\n"

    train_tasks = json_task['train']
    test_task = json_task['test']

    for i, task in enumerate(train_tasks):
        final_output += f"Example {i + 1}: Input\n"

        # Convert input grid to the new format with color words
        for row in task['input']:
            row_str = '|'.join(number_to_color[num] for num in row)
            final_output += f"{row_str}\n"

        final_output += f"\nExample {i + 1}: Output\n"

        # Convert output grid to the new format with color words
        for row in task['output']:
            row_str = '|'.join(number_to_color[num] for num in row)
            final_output += f"{row_str}\n"

        final_output += "\n"

    final_output += "Test Input\n"

    # Convert test input grid to the new format with color words
    for row in test_task[test_input_index]['input']:
        row_str = '|'.join(number_to_color[num] for num in row)
        final_output += f"{row_str}\n"

    final_output += "\nYour Response:"

    return final_output

def parse_prediction(prediction_string: str) -> List[List[int]]:
    # Split the string into lines
    lines = prediction_string.strip().split('\n')
    prediction = []
    grid_started = False  # Flag to indicate if grid lines have started
    for line in lines:
        if line.lower().startswith('analysis') or not line.strip():
            # Skip analysis lines or empty lines
            continue
        if any(color in line for color in color_to_number):  # Check if the line contains grid-like information
            grid_started = True
        if grid_started:
            # Parse grid-like rows
            row_colors = line.strip().split('|')
            row = []
            for color in row_colors:
                color = color.strip().lower()
                if color in color_to_number:
                    row.append(color_to_number[color])
                else:
                    raise ValueError(f"Unknown color '{color}' in prediction.")
            prediction.append(row)
    return prediction

def parse_prediction_with_analysis(prediction_string: str) -> Tuple[str, List[List[int]]]:
    """
    Parse the prediction string to extract both the analysis of transformation and the final grid prediction.

    Args:
        prediction_string (str): The response from the LLM.

    Returns:
        Tuple[str, List[List[int]]]: A tuple where the first element is the analysis text and
                                     the second element is the parsed grid prediction.
    """
    # Split the string into lines
    lines = prediction_string.strip().split('\n')
    analysis = ""
    grid_started = False  # Flag to indicate if grid lines have started
    prediction = []

    # Indicators for analysis and grid start/end markers
    analysis_start_marker = "---Analysis Start---"
    grid_start_marker = "---Output Grid Start---"
    grid_end_marker = "---Output Grid End---"

    for line in lines:
        # Remove any extra whitespace
        line = line.strip()
        
        # Check for markers
        if line == analysis_start_marker:
            # Analysis marker encountered, begin recording analysis
            analysis = ""
            continue
        elif line == grid_start_marker:
            # Grid start marker encountered, switch to grid parsing mode
            grid_started = True
            continue
        elif line == grid_end_marker:
            # Grid end marker encountered, stop grid parsing
            grid_started = False
            break
        elif not grid_started:
            # If we are still in the analysis section, keep recording analysis
            analysis += line + "\n"
        elif grid_started:
            # We are in grid parsing mode, so parse the grid-like rows
            row_colors = line.split('|')
            row = []
            # Check if each element in the row corresponds to a valid color
            for color in row_colors:
                color = color.strip().lower()
                if color in color_to_number:
                    row.append(color_to_number[color])
                else:
                    # Skip any non-color line that might have been included erroneously
                    row = []  # Invalidate the row if an unexpected line is encountered
                    break
            if row:  # Append only valid rows to the prediction
                prediction.append(row)
    
    return analysis.strip(), prediction



def get_task_prediction(challenge_tasks, solutions, logger, task_id, test_input_index) -> List[List]:

    # CoT example to be added at the beginning of each prompt
    cot_string = """Sample grid tranformation:

Input grid

black, black, black, black, black, black, black, black, black
black, black, black, black, yellow, black, black, black, black
black, black, yellow, yellow, black, black, black, black, black
black, black, yellow, black, black, black, black, black, black
black, black, black, black, black, black, black, black, black
black, black, black, black, black, black, black, black, black
black, black, black, black, black, yellow, black, black, black
black, black, black, black, black, yellow, yellow, yellow, black
black, black, black, black, black, black, yellow, black, black


Output grid

black, black, black, black, black, black, black, black, black
black, black, orange, orange, yellow, black, black, black, black
black, black, yellow, yellow, orange, black, black, black, black 
black, black, yellow, orange, orange, black, black, black, black
black, black, black, black, black, black, black, black, black
black, black, black, black, black, black, black, black, black
black, black, black, black, black, yellow, orange, orange, black 
black, black, black, black, black, yellow, yellow, yellow, black 
black, black, black, black, black, orange, yellow, orange, black


Analysis of transformation: Add orange colored cells in locations adjacent to yellow colored cells so that together they form three by three squares.
End of sample. 

Beginning of your task:
"""

    # Get the string representation of the task
    task_string = json_task_to_string(challenge_tasks, task_id, test_input_index)

    # Combine the CoT example and task string into the prompt
    context_prompt = cot_string + "\n" + task_string

    # Prompt template 1
    prompt = PromptTemplate(
        template="You are a chatbot with human-like reasoning and abstraction capabilities.\n"
                 "We will engage in tasks that require reasoning and logic.\n"
                 "You will be presented with grids of colored cells. Black color representes the background and the other colors represent objects on the grid.\n"
                 "Follow these steps:\n"
                 "1. You will first receive a sample grid transformation where the input and output grids are shown as well as an analysis of the transformation.\n"
                 "2. You will then be presented with a novel task.\n"
                 "3. Carefully analyze each input-output example in the task and identify the transformation.\n"
                 "4. Describe the transformation step-by-step.\n"
                 "5. Use the marker '---Analysis Start---' before providing your analysis of the transformation.\n"                
                 "6. Apply the identified transformation to the Test input grid to generate the output grid.\n"
                 "7. Use the marker '---Output Grid Start---' before providing the final output grid.\n"
                 "8. Use the same format as the one provided in the examples for your output grid.\n"
                 "9. Use the marker '---Output Grid End---' at the end of the final output grid.\n"
                 "\n\n{context_prompt}\n",
        input_variables=["context_prompt"]
    )

    # Generate the full prompt
    formatted_prompt = prompt.format(context_prompt=context_prompt)

    # Log the prompt
    logger.info(f"Prompt:\n{formatted_prompt}")

    # Call the model and get the prediction
    response = llm.invoke(formatted_prompt)

    # Check if the response content is empty
    if not response.content.strip():
        logger.error(f"Empty response received from LLM for Task ID {task_id}, Test Input Index {test_input_index}")
        return []  # Return an empty list if the response is empty
    
    # Log the raw LLM response for debugging
    logger.info(f"Raw LLM Response: {response.content}")
    
     # Extract the actual prediction and analysis from the response content
    analysis, prediction = parse_prediction_with_analysis(response.content)
    
    # Log the analysis of transformation separately
    logger.info(f"Analysis of transformations for Task ID {task_id}, Test Input Index {test_input_index}:\n{analysis}\n")
    
    # Also log the correct solution
    correct_solution = solutions[task_id][test_input_index]  # Get the correct solution
    # Convert the correct solution to the new format for logging
    solution_lines = ['|'.join(number_to_color[num] for num in row) for row in correct_solution]
    solution_string = '\n'.join(solution_lines)
    logger.info(f"Solution:\n{solution_string}\n")

    return prediction

def run_model(challenges, solutions, logger, NUM_ATTEMPTS=1, RETRY_ATTEMPTS=3, NUM_TASKS=None):

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
                                                         logger=logger,
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

def score_results(results, solutions, logger) -> Tuple[float, int]:
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
            llm = ChatOpenAI(model=model_name, api_key=OPENAI_API_KEY, max_tokens=3000, temperature=0.0)
            break
        elif model_choice == "5":
            model_name = "o1-preview"
            llm = ChatOpenAI(model=model_name, api_key=OPENAI_API_KEY, max_tokens=3000, temperature=0.0)
            break
        elif model_choice == "6":
            model_name = "claude-3-5-sonnet-20240620"
            llm = ChatAnthropic(model=model_name, api_key=ANTHROPIC_API_KEY, max_tokens=3000, temperature=0.0)
            break
        else:
            print("Invalid input. Please enter 1, 2, 3, 4, 5, or 6.")    

    # Generate log file name based on the selected model and timestamp
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
    test_results = run_model(challenges, solutions, logger, NUM_TASKS=NUM_TASKS)

    # Score the results
    score_result = score_results(results=test_results, solutions=solutions, logger=logger)

    logger.info(f"Model name: {model_name}, Model temperature: {llm.temperature}")
    logger.info(f"Final score: {score_result['total_score']} of {score_result['total_tasks_scored']} "
          f"({round(score_result['total_score'] / score_result['total_tasks_scored'] * 100, 2)}%)")

# Start the program
if __name__ == "__main__":
    main(task_set='training')