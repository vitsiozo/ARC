import json
import networkx as nx
from networkx.algorithms.components import connected_components
from itertools import combinations
import os

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

def main():
    # Step 1: Prompt the user for the task_id.json filename
    task_id_file = input("Enter the task ID filename (e.g., 'abcd1234.json'): ")

    # Base directory where the task files are located
    task_base_dir = "/Users/vitsiozo/Desktop/MSc AI/Modules/Project/ARC/datasets/subset50/"

    # Construct the full path to the task file
    task_file = os.path.join(task_base_dir, task_id_file)

    # Check if the task file exists
    if not os.path.exists(task_file):
        print(f"Task file '{task_file}' not found.")
        return

    # Load the task data
    with open(task_file, 'r') as f:
        task_data = json.load(f)

    # Extract task_id from the filename (without '.json' extension)
    task_id = os.path.splitext(task_id_file)[0]

    # Path to the solutions directory
    solutions_dir = "/Users/vitsiozo/Desktop/MSc AI/Modules/Project/ARC/datasets/subset50_ARGA_solutions"
    solutions_file = os.path.join(solutions_dir, f"solutions_{task_id}.json")

    # Check if the solutions file exists
    if os.path.exists(solutions_file):
        with open(solutions_file, 'r') as f:
            solutions_data = json.load(f)
        abstraction_method_name = solutions_data.get("abstraction", "nbccg")
        print(f"Using abstraction method '{abstraction_method_name}' from solutions file.")
    else:
        abstraction_method_name = "nbccg"
        print(f"Solutions file not found. Defaulting to abstraction method '{abstraction_method_name}'.")

    # Ensure the abstraction method exists
    if abstraction_method_name not in Image.abstraction_ops:
        print(f"Abstraction method '{abstraction_method_name}' not recognized. Defaulting to 'nbccg'.")
        abstraction_method_name = "nbccg"

    abstraction_method = Image.abstraction_ops[abstraction_method_name]

    # Prompt the user to choose the encoding
    encoding_options = ["object_json", "object_descriptor"]
    print("\nAvailable encodings:")
    for idx, enc in enumerate(encoding_options, start=1):
        print(f"{idx}. {enc}")
    try:
        encoding_choice = int(input(f"Choose an encoding (1-{len(encoding_options)}): "))
        if encoding_choice not in range(1, len(encoding_options) + 1):
            raise ValueError
        encoding = encoding_options[encoding_choice - 1]
    except ValueError:
        print("Invalid choice. Defaulting to 'object_json'.")
        encoding = "object_json"

    # Step 2: Generate abstract representations
    images = []

    # Process training examples
    for idx, pair in enumerate(task_data['train']):
        input_image = Image(grid=pair['input'], name=f'train_input_{idx+1}')
        output_image = Image(grid=pair['output'], name=f'train_output_{idx+1}')
        # Call the selected abstraction method
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

    # Step 3: Create string or JSON encodings
    for idx, (input_image, output_image) in enumerate(images):
        print(f'\n--- Pair {idx+1} ---')
        print(f'Input Image ({input_image.name}):')
        print(f'Image size: {input_image.image_size}')
        print('Objects:')
        input_encoding = input_image.get_graph_encoded_string(encoding=encoding)
        print(input_encoding)

        if output_image:
            print(f'Output Image ({output_image.name}):')
            print(f'Image size: {output_image.image_size}')
            print('Objects:')
            output_encoding = output_image.get_graph_encoded_string(encoding=encoding)
            print(output_encoding)

if __name__ == "__main__":
    main()
