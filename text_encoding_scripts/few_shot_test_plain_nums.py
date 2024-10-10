import os
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
log_dir = '/Users/vitsiozo/Desktop/MSc AI/Modules/Project/ARC/log_txt_output/few_shot/numbers_plain'

# Directory where tasks are stored
task_sets = {
    'training': {
        'challenges': '/Users/vitsiozo/Desktop/MSc AI/Modules/Project/ARC/datasets/50_challenges.json',
        'solutions': '/Users/vitsiozo/Desktop/MSc AI/Modules/Project/ARC/datasets/50_solutions.json',
    }
}

# Function to load ARC tasks from JSON files
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

        # Convert input grid to the new format
        for row in task['input']:
            row_str = ''.join(str(num) for num in row)
            final_output += f"{row_str}\n"

        final_output += f"\nExample {i + 1}: Output\n"

        # Convert output grid to the new format
        for row in task['output']:
            row_str = ''.join(str(num) for num in row)
            final_output += f"{row_str}\n"

        final_output += "\n"

    final_output += "Test Input\n"

    # Convert test input grid to the new format
    for row in test_task[test_input_index]['input']:
        row_str = ''.join(str(num) for num in row)
        final_output += f"{row_str}\n"

    final_output += "\nYour Response:"

    return final_output

# Function to parse the model's response into a grid format
def parse_prediction(prediction_string: str) -> List[List[int]]:
    # Split the string into lines
    lines = prediction_string.strip().split('\n')
    # For each line, convert the string of digits into a list of integers
    prediction = []
    for line in lines:
        if line.strip() == '':
            continue  # Skip empty lines
        row = [int(char) for char in line.strip()]
        prediction.append(row)
    return prediction

# Function to get a prediction for a single ARC task
def get_task_prediction(challenge_tasks, solutions, logger, task_id, test_input_index) -> List[List]:

    # Get the string representation of the task
    task_string = json_task_to_string(challenge_tasks, task_id, test_input_index)

    # Prompt template 1
    prompt = PromptTemplate(
        template="You are a chatbot with human-like reasoning and abstraction capabilities. "
                 "We will engage in tasks that require reasoning and logic. "
                 "Each task will demonstrate a transformation from an input to an output grid. "
                 "For each task, you will receive a few examples that demonstrate the transformation from input to output. "
                 "After the examples you'll receive a new input grid called Test. "                
                 "Your task is to determine the corresponding output grid from the transformation you are able to infer from the examples. "
                 "Use the same format as the one provided in the examples for your answer. "
                 "Do not give any justification for your answer, just provide the output grid. "
                 "\n\n{task_string}\n",
        input_variables=["task_string"]
    )

    # Generate the full prompt
    formatted_prompt = prompt.format(task_string=task_string)

    # Log the prompt
    logger.info(f"Prompt:\n{formatted_prompt}")

    # Call the model and get the prediction
    response = llm.invoke(formatted_prompt)

    # Log the raw LLM response for debugging
    logger.info(f"Raw LLM Response: {response.content}")

    # Check if the response content is empty
    if not response.content.strip():
        logger.error(f"Empty response received from LLM for Task ID {task_id}, Test Input Index {test_input_index}")
        return []  # Return an empty list if the response is empty

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
    solution_string = "\n".join([''.join(map(str, row)) for row in correct_solution])  # Format the solution as a string
    logger.info(f"Solution:\n{solution_string}\n")

    return prediction

# Function to run the model on ARC tasks
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

# Function to score the results of the ARC tasks
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

# Main function that prompts for model and runs the tasks
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
