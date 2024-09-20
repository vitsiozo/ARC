# abstraction.py

import os
import json
import networkx as nx
from networkx.algorithms.components import connected_components
from itertools import combinations

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
        # Include other abstraction methods as needed
    }

    def __init__(self, grid, name):
        self.name = name
        self.grid = grid
        self.height = len(grid)
        self.width = len(grid[0]) if self.height > 0 else 0
        self.image_size = (self.height, self.width)
        self.background_color = self._determine_background_color()
        self.abstracted_graph = None

    def _determine_background_color(self):
        # Determine the most common color (background color)
        colors = [color for row in self.grid for color in row]
        return max(set(colors), key=colors.count) if colors else 0

    def get_2d_grid_graph(self, top_down=True):
        graph = nx.grid_2d_graph(self.height, self.width)
        for r in range(self.height):
            for c in range(self.width):
                graph.nodes[(r, c)]['color'] = self.grid[r][c]
        return graph

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

        self.abstracted_graph = non_black_components_graph
        self.abstraction = "nbccg"

    # Include other abstraction methods as necessary

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
        else:
            raise ValueError(f"Encoding '{encoding}' not supported.")

def get_abstraction_method(task_id):
    """
    Determines the abstraction method to use for a given task_id.
    """
    # Path to the solutions directory
    solutions_dir = "/path/to/your/solutions_directory"  # Update this path
    solutions_file = os.path.join(solutions_dir, f"solutions_{task_id}.json")

    # Check if the solutions file exists
    if os.path.exists(solutions_file):
        with open(solutions_file, 'r') as f:
            solutions_data = json.load(f)
        abstraction_method_name = solutions_data.get("abstraction", "nbccg")
        print(f"Using abstraction method '{abstraction_method_name}' from solutions file.")
    else:
        abstraction_method_name = "nbccg"
        print(f"Solutions file not found for task {task_id}. Defaulting to abstraction method '{abstraction_method_name}'.")

    # Ensure the abstraction method exists
    if abstraction_method_name not in Image.abstraction_ops:
        print(f"Abstraction method '{abstraction_method_name}' not recognized. Defaulting to 'nbccg'.")
        abstraction_method_name = "nbccg"

    return abstraction_method_name

def generate_abstracted_task(task_data, task_id):
    """
    Generates the abstracted representation of a task using the appropriate abstraction method.
    """
    abstraction_method_name = get_abstraction_method(task_id)
    encoding = "object_json"  # Assuming we are using object_json encoding

    images = []

    # Process training examples
    for idx, pair in enumerate(task_data['train']):
        input_image = Image(grid=pair['input'], name=f'train_input_{idx+1}')
        output_image = Image(grid=pair['output'], name=f'train_output_{idx+1}')
        # Call the selected abstraction method
        abstraction_method = Image.abstraction_ops[abstraction_method_name]
        getattr(input_image, abstraction_method)()
        getattr(output_image, abstraction_method)()
        images.append((input_image, output_image))

    # Process test examples
    for idx, pair in enumerate(task_data.get('test', [])):
        input_image = Image(grid=pair['input'], name=f'test_input_{idx+1}')
        getattr(input_image, abstraction_method)()
        output_image = None
        if 'output' in pair:
            output_image = Image(grid=pair['output'], name=f'test_output_{idx+1}')
            getattr(output_image, abstraction_method)()
        images.append((input_image, output_image))

    # Create string or JSON encodings
    abstracted_task_str = ""
    for idx, (input_image, output_image) in enumerate(images):
        abstracted_task_str += f'\n--- Pair {idx+1} ---\n'
        abstracted_task_str += f'Input Image ({input_image.name}):\n'
        abstracted_task_str += f'Image size: {input_image.image_size}\n'
        abstracted_task_str += 'Objects:\n'
        input_encoding = input_image.get_graph_encoded_string(encoding=encoding)
        abstracted_task_str += f'{input_encoding}\n'

        if output_image:
            abstracted_task_str += f'Output Image ({output_image.name}):\n'
            abstracted_task_str += f'Image size: {output_image.image_size}\n'
            abstracted_task_str += 'Objects:\n'
            output_encoding = output_image.get_graph_encoded_string(encoding=encoding)
            abstracted_task_str += f'{output_encoding}\n'

    return abstracted_task_str
