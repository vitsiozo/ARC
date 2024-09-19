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

    def get_2d_grid_graph(self):
        graph = nx.grid_2d_graph(self.height, self.width)
        for r in range(self.height):
            for c in range(self.width):
                graph.nodes[(r, c)]['color'] = self.grid[r][c]
        return graph

    def get_non_black_components_graph(self):
        """
        Generates a graph where each node represents a connected component of non-black pixels.
        """
        graph = self.get_2d_grid_graph()
        component_graph = nx.Graph()
        node_id = 1

        # Identify connected components for each color (excluding background color 0)
        for color in range(1, 10):
            color_nodes = [n for n, attr in graph.nodes(data=True) if attr['color'] == color]
            subgraph = graph.subgraph(color_nodes)
            components = list(connected_components(subgraph))

            for component in components:
                component_graph.add_node(
                    node_id,
                    coordinates=sorted(component),
                    color=color,
                    size=len(component)
                )
                node_id += 1

        # Optionally, add edges between components (not required for encoding)
        self.abstracted_graph = component_graph

    def get_graph_encoded_string(self, encoding="object_json"):
        """
        Returns a string or JSON representation of the abstracted graph suitable for inclusion in prompts.
        """
        if not self.abstracted_graph:
            raise ValueError("Abstracted graph is not generated. Call get_non_black_components_graph() first.")

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

def main():
    # Step 1: Load the task file
    task_file = input("Enter the path to the task file (e.g., 'path/to/task.json'): ")
    with open(task_file, 'r') as f:
        task_data = json.load(f)

    # Step 2: Generate abstract representations
    images = []

    # Process training examples
    for idx, pair in enumerate(task_data['train']):
        input_image = Image(grid=pair['input'], name=f'train_input_{idx+1}')
        output_image = Image(grid=pair['output'], name=f'train_output_{idx+1}')
        input_image.get_non_black_components_graph()
        output_image.get_non_black_components_graph()
        images.append((input_image, output_image))

    # Process test examples
    for idx, pair in enumerate(task_data.get('test', [])):
        input_image = Image(grid=pair['input'], name=f'test_input_{idx+1}')
        input_image.get_non_black_components_graph()
        output_image = None
        if 'output' in pair:
            output_image = Image(grid=pair['output'], name=f'test_output_{idx+1}')
            output_image.get_non_black_components_graph()
        images.append((input_image, output_image))

    # Step 3: Create string or JSON encodings
    for idx, (input_image, output_image) in enumerate(images):
        print(f'\n--- Pair {idx+1} ---')
        print(f'Input Image ({input_image.name}):')
        input_encoding = input_image.get_graph_encoded_string(encoding='object_json')
        print(input_encoding)

        if output_image:
            print(f'\nOutput Image ({output_image.name}):')
            output_encoding = output_image.get_graph_encoded_string(encoding='object_json')
            print(output_encoding)

if __name__ == "__main__":
    main()
