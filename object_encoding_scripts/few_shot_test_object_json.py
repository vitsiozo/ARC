import os
import json
import re
import logging
from datetime import datetime
from typing import List, Tuple
from langchain_openai import ChatOpenAI  # To work with OpenAI
from langchain.prompts import PromptTemplate  # To help create our prompt
import networkx as nx
from networkx.algorithms.components import connected_components
from itertools import combinations

# Get API key for ChatGPT
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Choose the model to use
model_name = 'gpt-4o-mini'
llm = ChatOpenAI(model_name=model_name, openai_api_key=OPENAI_API_KEY, max_tokens=3000, temperature=0.0)

# Directory for logging files
log_dir = '/Users/vitsiozo/Desktop/MSc AI/Modules/Project/ARC/log_obj_output'

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

# --- Begin of the Image Class and Abstraction Methods ---

class Image:
    """
    Represents an input or output image in the ARC dataset.
    """

    # Mapping from digits to words (if needed for encoding)
    digit_to_word = {
        0: 'black', 1: 'blue', 2: 'red', 3: 'green',
        4: 'yellow', 5: 'grey', 6: 'fuchsia', 7: 'orange',
        8: 'teal', 9: 'brown'
    }

    # Dictionary mapping abstraction names to methods
    abstraction_ops = {
        "nbccg": "get_non_black_components_graph",
        "nbccg_d": "get_non_black_components_with_diagonals_graph",
        "ccgbr": "get_connected_components_graph_background_removed",
        "ccgbr2": "get_connected_components_graph_background_removed_2",
        "ccg": "get_connected_components_graph",
        "mcccg": "get_multicolor_connected_components_graph",
        "mcccg_d": "get_multicolor_components_with_diagonals_graph",
        "na": "get_no_abstraction_graph",
        "nbvcg": "get_non_background_vertical_connected_components_graph",
        "nbvcg2": "get_non_background_vertical_connected_components_graph_2",
        "nbvcg3": "get_non_background_vertical_components_graph",
        "nbhcg": "get_non_background_horizontal_connected_components_graph",
        "nbhcg2": "get_non_background_horizontal_connected_components_graph_2",
        "nbhcg3": "get_non_background_horizontal_components_graph",
        "lrg": "get_largest_rectangle_graph"
    }

    def __init__(self, grid, name):
        self.name = name
        self.grid = grid
        self.height = len(grid)
        self.width = len(grid[0]) if self.height > 0 else 0
        self.image_size = (self.height, self.width)
        self.background_color = self._determine_background_color()
        self.abstracted_graph = None
        self.corners = {
            (0, 0),
            (0, self.width - 1),
            (self.height - 1, 0),
            (self.height - 1, self.width - 1)
        }

    def _determine_background_color(self):
        # Determine the most common color (background color)
        colors = [color for row in self.grid for color in row]
        return max(set(colors), key=colors.count) if colors else 0

    def get_2d_grid_graph(self, top_down=True):
        if top_down:
            graph = nx.grid_2d_graph(self.height, self.width)
            for r in range(self.height):
                for c in range(self.width):
                    graph.nodes[(r, c)]['color'] = self.grid[r][c]
        else:
            graph = nx.grid_2d_graph(self.height, self.width)
            for r in range(self.height):
                for c in range(self.width):
                    graph.nodes[(self.height - r - 1, c)]['color'] = self.grid[r][c]
        return graph

    # Include all the abstraction methods from your original code with correct indentation
    # For brevity, I'm including only a couple of methods as examples

    def get_non_black_components_graph(self, graph=None):
        if not graph:
            graph = self.get_2d_grid_graph()

        non_black_components_graph = nx.Graph()

        node_id = 1
        for color in range(1, 10):  # Colors 1 to 9 (excluding black which is 0)
            color_nodes = (node for node, data in graph.nodes(data=True) if data.get("color") == color)
            color_subgraph = graph.subgraph(color_nodes)
            color_connected_components = connected_components(color_subgraph)
            for component in color_connected_components:
                non_black_components_graph.add_node(
                    node_id,
                    coordinates=sorted(component),
                    color=color,
                    size=len(component)
                )
                node_id += 1

        for node_1, node_2 in combinations(non_black_components_graph.nodes, 2):
            nodes_1 = non_black_components_graph.nodes[node_1]["coordinates"]
            nodes_2 = non_black_components_graph.nodes[node_2]["coordinates"]
            for n1 in nodes_1:
                for n2 in nodes_2:
                    if n1[0] == n2[0]:  # Same row
                        if all(graph.nodes[n1[0], c]["color"] == 0 for c in range(min(n1[1], n2[1]) + 1, max(n1[1], n2[1]))):
                            non_black_components_graph.add_edge(node_1, node_2, direction="horizontal")
                            break
                    elif n1[1] == n2[1]:  # Same column
                        if all(graph.nodes[r, n1[1]]["color"] == 0 for r in range(min(n1[0], n2[0]) + 1, max(n1[0], n2[0]))):
                            non_black_components_graph.add_edge(node_1, node_2, direction="vertical")
                            break
                else:
                    continue
                break

        self.abstracted_graph = non_black_components_graph
        self.abstraction = "nbccg"

    # [Include the rest of your abstraction methods here with correct indentation]

    def get_graph_encoded_string(self, encoding="object_json"):
        """
        Returns a string or JSON representation of the abstracted graph suitable for inclusion in prompts.
        """
        if not self.abstracted_graph:
            raise ValueError("Abstracted graph is not generated. Call an abstraction method first.")

        if encoding == "object_json":
            nodes = []
            for node_id, attrs in self.abstracted_graph.nodes(data=True):
                node_info = {
                    "coordinates": attrs["coordinates"],
                    "color": attrs["color"],
                    "size": attrs["size"]
                }
                nodes.append(node_info)
            return json.dumps(nodes, indent=2)
        elif encoding == "object_descriptor":
            node_list = ""
            for node_id, attrs in self.abstracted_graph.nodes(data=True):
                coordinates_str = ', '.join(f'({r}, {c})' for r, c in attrs["coordinates"])
                node_str = f'Object {node_id}: coordinates=[{coordinates_str}], color={attrs["color"]}, size={attrs["size"]}\n'
                node_list += node_str
            return node_list
        else:
            raise ValueError(f"Encoding '{encoding}' not supported.")
        
# --- End of the Image Class and Abstraction Methods ---

def get_abstraction_method(task_id):
    # Path to the solutions directory
    solutions_dir = "/Users/vitsiozo/Desktop/MSc AI/Modules/Project/ARC/datasets/subset50_ARGA_solutions"
    solutions_file = os.path.join(solutions_dir, f"solutions_{task_id}.json")

    # Check if the solutions file exists
    if os.path.exists(solutions_file):
        with open(solutions_file, 'r') as f:
            solutions_data = json.load(f)
        abstraction_method_name = solutions_data.get("abstraction", "nbccg")
        logger.info(f"Using abstraction method '{abstraction_method_name}' from solutions file.")
    else:
        abstraction_method_name = "nbccg"
        logger.info(f"Solutions file not found for task {task_id}. Defaulting to abstraction method '{abstraction_method_name}'.")

    # Ensure the abstraction method exists
    if abstraction_method_name not in Image.abstraction_ops:
        logger.info(f"Abstraction method '{abstraction_method_name}' not recognized. Defaulting to 'nbccg'.")
        abstraction_method_name = "nbccg"

    return abstraction_method_name

def get_abstracted_representation(grid, name, abstraction_method_name):
    image = Image(grid=grid, name=name)
    abstraction_method = Image.abstraction_ops[abstraction_method_name]
    getattr(image, abstraction_method)()
    abstracted_representation = image.get_graph_encoded_string(encoding="object_json")
    return abstracted_representation

def get_task_abstracted_string(challenge_tasks: dict, task_id: str, test_input_index: int) -> str:
    json_task = challenge_tasks[task_id]

    train_tasks = json_task['train']
    test_task = json_task['test']

    # Get the abstraction method for this task
    abstraction_method_name = get_abstraction_method(task_id)

    # Build the task string
    final_output = "Training Examples\n\n"

    for i, task in enumerate(train_tasks):
        # Generate abstracted representations for input and output
        input_abstracted = get_abstracted_representation(
            grid=task['input'],
            name=f'train_input_{i+1}',
            abstraction_method_name=abstraction_method_name
        )
        output_abstracted = get_abstracted_representation(
            grid=task['output'],
            name=f'train_output_{i+1}',
            abstraction_method_name=abstraction_method_name
        )
        final_output += f"Example {i + 1}: Input\n{input_abstracted}\n\n"
        final_output += f"Example {i + 1}: Output\n{output_abstracted}\n\n"

    # For the test input
    test_input_abstracted = get_abstracted_representation(
        grid=test_task[test_input_index]['input'],
        name=f'test_input_{test_input_index+1}',
        abstraction_method_name=abstraction_method_name
    )

    final_output += "Test Input\n"
    final_output += f"{test_input_abstracted}\n\nYour Response:"

    return final_output

def get_task_prediction(challenge_tasks, solutions, task_id, test_input_index) -> List[List]:
    # Get the string representation of the task
    task_string = get_task_abstracted_string(challenge_tasks, task_id, test_input_index)

    # Prompt template
    prompt = PromptTemplate(
        template="You are a reasoning assistant adept at solving abstract reasoning tasks. "
                 "Given the abstracted representations of training examples (inputs and outputs), "
                 "infer the transformation and apply it to the test input to produce the output grid. "
                 "Provide the output grid as a list of lists of integers. "
                 "Do not provide a justification for the answer, just the answer. \n\n{task_string}\n",
        input_variables=["task_string"]
    )

    # Generate the full prompt
    formatted_prompt = prompt.format(task_string=task_string)

    # Log the prompt
    logger.info(f"Prompt:\n{formatted_prompt}")

    # Call the model and get the prediction
    response = llm.predict(formatted_prompt)

    # Extract the actual prediction from the response
    prediction_string = response.strip()

    # Parse the string as JSON
    try:
        prediction = json.loads(prediction_string)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        prediction = []  # Assign an empty list if parsing fails

    # Let's find the shape of our prediction
    num_rows = len(prediction)
    num_cols = len(prediction[0]) if num_rows > 0 else 0
    # Log the prediction and the grid size
    logger.info(f"   *** Prediction for Task ID {task_id}, Test Input Index {test_input_index}")
    logger.info(f"       Grid Size: {num_rows}x{num_cols}\n{prediction_string}\n")

    # Also log the correct solution
    correct_solution = solutions[task_id][test_input_index]  # Get the correct solution
    solution_string = "\n".join([str(row) for row in correct_solution])  # Format the solution as a string
    logger.info(f"   *** Solution\n{solution_string}\n")

    return prediction

def run_model(challenges, solutions, NUM_ATTEMPTS=2, RETRY_ATTEMPTS=3, NUM_TASKS=None):
    # A dict to hold the results returned after all predictions are made
    results = {}

    # Run through each task  
    for i, task_id in enumerate(challenges):
        task_attempts = []  # List to store all attempts for the current task

        # Go through each test pair to get a prediction. 96% of challenges have just 1 pair.
        for t, pair in enumerate(challenges[task_id]['test']):
            logger.info(f"Starting task #{i + 1} ({task_id}), pair #{t+1}")

            # Dictionary to store attempts for the current test pair
            pair_attempts = {}

            # Run through each prediction attempt
            for attempt in range(1, NUM_ATTEMPTS + 1):
                attempt_key = f"attempt_{attempt}"
                pair_attempts[attempt_key] = []  # Init your attempt

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
                            pair_attempts[attempt_key] = []  # Assign empty list if all retries fail

            # After you get your attempts, append them to the task attempts
            task_attempts.append(pair_attempts)

        # Append the task attempts to the results with the task_id as the key
        results[task_id] = task_attempts

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
                    break  # If it is correct, log it and break the loop

            if pair_correct:
                task_score += 1

        task_score /= num_pairs
        total_score += task_score

    logger.info(f"Total score: {total_score}, Total tasks scored: {total_tasks}")
    return {
        'total_score': total_score,
        'total_tasks_scored': total_tasks
    }

def main(task_set='training'):
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
