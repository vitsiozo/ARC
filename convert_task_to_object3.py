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
        self.corners = {(0, 0), (0, self.width - 1), (self.height - 1, 0), (self.height - 1, self.width - 1)}

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

    # Include all the abstraction methods from the original code
    def get_non_black_components_graph(self, graph=None):

        if not graph:
            graph = self.get_2d_grid_graph()

        non_black_components_graph = nx.Graph()

        # for color in self.colors_included:
        node_id = 1
        for color in range(10):
            if color == 0:
                continue
            color_nodes = (node for node, data in graph.nodes(data=True) if data.get("color") == color)
            color_subgraph = graph.subgraph(color_nodes)
            color_connected_components = connected_components(color_subgraph)
            for i, component in enumerate(color_connected_components):
                # non_black_components_graph.add_node((color, i), pixels=list(component), color=color, size=len(list(component)))
                non_black_components_graph.add_node(node_id, coordinates=sorted(list(component)), color=color, size=len(list(component)))
                node_id += 1
        for node_1, node_2 in combinations(non_black_components_graph.nodes, 2):
            nodes_1 = non_black_components_graph.nodes[node_1]["coordinates"]
            nodes_2 = non_black_components_graph.nodes[node_2]["coordinates"]
            for n1 in nodes_1:
                for n2 in nodes_2:
                    if n1[0] == n2[0]:  # two nodes on the same row
                        for column_index in range(min(n1[1], n2[1])+1, max(n1[1], n2[1])):
                            if graph.nodes[n1[0], column_index]["color"] != 0:
                                break
                        else:
                            non_black_components_graph.add_edge(node_1, node_2, direction="horizontal")
                            break
                    elif n1[1] == n2[1]:  # two nodes on the same column:
                        for row_index in range(min(n1[0], n2[0])+1, max(n1[0], n2[0])):
                            if graph.nodes[row_index, n1[1]]["color"] != 0:
                                break
                        else:
                            non_black_components_graph.add_edge(node_1, node_2, direction="vertical")
                            break
                else:
                    continue
                break

        self.abstracted_graph = non_black_components_graph
        self.abstraction = "nbccg"
        # return ARCGraph(non_black_components_graph, self.name, self, "nbccg")

    def get_non_black_components_with_diagonals_graph(self, graph=None):

        if not graph:
            graph = self.get_2d_grid_graph()

        non_black_components_graph = nx.Graph()

        with_diagonals = graph.copy()
        with_diagonals.add_edges_from([((x, y), (x + 1, y + 1)) for x in range(self.height-1) for y in range(self.width-1)]
                                      + [((x + 1, y), (x, y + 1)) for x in range(self.height-1) for y in range(self.width-1)])

        node_id = 1
        for color in range(10):
            if color == 0:
                continue
            color_nodes = (node for node, data in with_diagonals.nodes(data=True) if data.get("color") == color)
            color_subgraph = with_diagonals.subgraph(color_nodes)
            color_connected_components = connected_components(color_subgraph)
            for i, component in enumerate(color_connected_components):
                non_black_components_graph.add_node(node_id, coordinates=list(component), color=color, size=len(list(component)))
                node_id += 1

        # for node_1, attributes_1 in color_connected_components_graph.nodes(data=True):
        #     for node_2, attributes_2 in color_connected_components_graph.nodes(data=True):
        for node_1, node_2 in combinations(non_black_components_graph.nodes, 2):
            nodes_1 = non_black_components_graph.nodes[node_1]["coordinates"]
            nodes_2 = non_black_components_graph.nodes[node_2]["coordinates"]
            for n1 in nodes_1:
                for n2 in nodes_2:
                    if n1[0] == n2[0]:  # two nodes on the same row
                        for column_index in range(min(n1[1], n2[1])+1, max(n1[1], n2[1])):
                            if with_diagonals.nodes[n1[0], column_index]["color"] != 0:
                                break
                        else:
                            non_black_components_graph.add_edge(node_1, node_2, direction="horizontal")
                            break
                    elif n1[1] == n2[1]:  # two nodes on the same column:
                        for row_index in range(min(n1[0], n2[0])+1, max(n1[0], n2[0])):
                            if with_diagonals.nodes[row_index, n1[1]]["color"] != 0:
                                break
                        else:
                            non_black_components_graph.add_edge(node_1, node_2, direction="vertical")
                            break
                else:
                    continue
                break

        self.abstracted_graph = non_black_components_graph
        self.abstraction = "nbccg_d"

    def get_connected_components_graph_background_removed(self, graph=None):
        if not graph:
            graph = self.get_2d_grid_graph()

        ccgbr = nx.Graph()

        # for color in self.colors_included:
        node_id = 1
        for color in range(10):
            color_nodes = (node for node, data in graph.nodes(data=True) if data.get("color") == color)
            color_subgraph = graph.subgraph(color_nodes)
            color_connected_components = connected_components(color_subgraph)
            if color != self.background_color:
                for i, component in enumerate(color_connected_components):
                    ccgbr.add_node(node_id, coordinates=sorted(list(component)), color=color, size=len(list(component)))
                    node_id += 1
            else:
                for i, component in enumerate(color_connected_components):
                    if len(set(component) & self.corners) == 0:  # background color + contains a corner
                        ccgbr.add_node(node_id, coordinates=sorted(list(component)), color=color, size=len(list(component)))
                        node_id += 1

        # for node_1, attributes_1 in color_connected_components_graph.nodes(data=True):
        #     for node_2, attributes_2 in color_connected_components_graph.nodes(data=True):
        for node_1, node_2 in combinations(ccgbr.nodes, 2):
            nodes_1 = ccgbr.nodes[node_1]["coordinates"]
            nodes_2 = ccgbr.nodes[node_2]["coordinates"]
            for n1 in nodes_1:
                for n2 in nodes_2:
                    if n1[0] == n2[0]:  # two nodes on the same row
                        for column_index in range(min(n1[1], n2[1]) + 1, max(n1[1], n2[1])):
                            if graph.nodes[n1[0], column_index]["color"] != self.background_color:
                                break
                        else:
                            ccgbr.add_edge(node_1, node_2, direction="horizontal")
                            break
                    elif n1[1] == n2[1]:  # two nodes on the same column:
                        for row_index in range(min(n1[0], n2[0]) + 1, max(n1[0], n2[0])):
                            if graph.nodes[row_index, n1[1]]["color"] != self.background_color:
                                break
                        else:
                            ccgbr.add_edge(node_1, node_2, direction="vertical")
                            break
                else:
                    continue
                break

        # return ARCGraph(ccgbr, self.name, self, "ccgbr")
        self.abstracted_graph = ccgbr
        self.abstraction = "ccgbr"

    def get_non_background_vertical_connected_components_graph(self, graph=None):
        if not graph:
            graph = self.get_2d_grid_graph()

        non_background_vertical_connected_components_graph = nx.Graph()

        # for color in self.colors_included:
        node_id = 1
        for color in range(10):
            color_connected_components = []
            if color == self.background_color:
                continue
            for column in range(self.width):
                color_nodes = (node for node, data in graph.nodes(data=True) if node[1] == column and data.get("color") == color)
                color_subgraph = graph.subgraph(color_nodes)
                color_connected_components.extend(list(connected_components(color_subgraph)))
            for i, component in enumerate(color_connected_components):
                non_background_vertical_connected_components_graph.add_node(node_id, coordinates=sorted(list(component)), color=color, size=len(list(component)))
                node_id += 1


        # for node_1, attributes_1 in color_connected_components_graph.nodes(data=True):
        #     for node_2, attributes_2 in color_connected_components_graph.nodes(data=True):
        for node_1, node_2 in combinations(non_background_vertical_connected_components_graph.nodes, 2):
            nodes_1 = non_background_vertical_connected_components_graph.nodes[node_1]["coordinates"]
            nodes_2 = non_background_vertical_connected_components_graph.nodes[node_2]["coordinates"]
            for n1 in nodes_1:
                for n2 in nodes_2:
                    if n1[0] == n2[0]:  # two nodes on the same row
                        for column_index in range(min(n1[1], n2[1])+1, max(n1[1], n2[1])):
                            if graph.nodes[n1[0], column_index]["color"] != self.background_color:
                                break
                        else:
                            non_background_vertical_connected_components_graph.add_edge(node_1, node_2, direction="horizontal")
                            break
                    elif n1[1] == n2[1]:  # two nodes on the same column:
                        for row_index in range(min(n1[0], n2[0])+1, max(n1[0], n2[0])):
                            if graph.nodes[row_index, n1[1]]["color"] != self.background_color:
                                break
                        else:
                            non_background_vertical_connected_components_graph.add_edge(node_1, node_2, direction="vertical")
                            break
                else:
                    continue
                break

        # return ARCGraph(non_background_vertical_connected_components_graph, self.name, self, "nbvcg")
        self.abstracted_graph = non_background_vertical_connected_components_graph
        self.abstraction = "nbvcg"

    def get_non_background_vertical_connected_components_graph_2(self, graph=None):
        if not graph:
            graph = self.get_2d_grid_graph()

        non_background_vertical_connected_components_graph = nx.Graph()

        # for color in self.colors_included:
        node_id = 1
        for color in range(10):
            color_connected_components = []
            for column in range(self.width):
                color_nodes = (node for node, data in graph.nodes(data=True) if node[1] == column and data.get("color") == color)
                color_subgraph = graph.subgraph(color_nodes)
                color_connected_components.extend(list(connected_components(color_subgraph)))
            for i, component in enumerate(color_connected_components):
                if color != self.background_color:
                    non_background_vertical_connected_components_graph.add_node(node_id, coordinates=sorted(list(component)), color=color, size=len(list(component)))
                    node_id += 1
                else:
                    component = list(component)
                    for node in component:
                        # if the node touches top or bottom of image it is not included
                        if node[0] == 0 or node[0] == self.height - 1:
                            break
                    else:
                        non_background_vertical_connected_components_graph.add_node(node_id, coordinates=component, color=color, size=len(component))
                        node_id += 1


        # for node_1, attributes_1 in color_connected_components_graph.nodes(data=True):
        #     for node_2, attributes_2 in color_connected_components_graph.nodes(data=True):
        for node_1, node_2 in combinations(non_background_vertical_connected_components_graph.nodes, 2):
            nodes_1 = non_background_vertical_connected_components_graph.nodes[node_1]["coordinates"]
            nodes_2 = non_background_vertical_connected_components_graph.nodes[node_2]["coordinates"]
            for n1 in nodes_1:
                for n2 in nodes_2:
                    if n1[0] == n2[0]:  # two nodes on the same row
                        for column_index in range(min(n1[1], n2[1])+1, max(n1[1], n2[1])):
                            if graph.nodes[n1[0], column_index]["color"] != self.background_color:
                                break
                        else:
                            non_background_vertical_connected_components_graph.add_edge(node_1, node_2, direction="horizontal")
                            break
                    elif n1[1] == n2[1]:  # two nodes on the same column:
                        for row_index in range(min(n1[0], n2[0])+1, max(n1[0], n2[0])):
                            if graph.nodes[row_index, n1[1]]["color"] != self.background_color:
                                break
                        else:
                            non_background_vertical_connected_components_graph.add_edge(node_1, node_2, direction="vertical")
                            break
                else:
                    continue
                break

        # return ARCGraph(non_background_vertical_connected_components_graph, self.name, self, "nbvcg")
        self.abstracted_graph = non_background_vertical_connected_components_graph
        self.abstraction = "nbvcg2"

    def get_non_background_vertical_components_graph(self, graph=None):
        if not graph:
            graph = self.get_2d_grid_graph(top_down=True)

        non_background_vertical_connected_components_graph = nx.Graph()

        # for color in self.colors_included:
        node_id = 1
        for column in range(self.width):

            # color_connected_components = []
            for color in range(10):
                if color == self.background_color:
                    continue
                color_nodes = (node for node, data in graph.nodes(data=True) if node[1] == column and data.get("color") == color)
                color_subgraph = graph.subgraph(color_nodes)
                # color_connected_components.append(list(color_subgraph.nodes()))
            # for i, component in enumerate(color_connected_components):
                component = list(color_subgraph.nodes())
                if not component:
                    continue
                non_background_vertical_connected_components_graph.add_node(node_id, coordinates=sorted(list(component)), color=color, size=len(list(component)))
                node_id += 1


        # for node_1, attributes_1 in color_connected_components_graph.nodes(data=True):
        #     for node_2, attributes_2 in color_connected_components_graph.nodes(data=True):
        for node_1, node_2 in combinations(non_background_vertical_connected_components_graph.nodes, 2):
            nodes_1 = non_background_vertical_connected_components_graph.nodes[node_1]["coordinates"]
            nodes_2 = non_background_vertical_connected_components_graph.nodes[node_2]["coordinates"]
            for n1 in nodes_1:
                for n2 in nodes_2:
                    if n1[0] == n2[0]:  # two nodes on the same row
                        for column_index in range(min(n1[1], n2[1])+1, max(n1[1], n2[1])):
                            if graph.nodes[n1[0], column_index]["color"] != self.background_color:
                                break
                        else:
                            non_background_vertical_connected_components_graph.add_edge(node_1, node_2, direction="horizontal")
                            break
                    elif n1[1] == n2[1]:  # two nodes on the same column:
                        for row_index in range(min(n1[0], n2[0])+1, max(n1[0], n2[0])):
                            if graph.nodes[row_index, n1[1]]["color"] != self.background_color:
                                break
                        else:
                            non_background_vertical_connected_components_graph.add_edge(node_1, node_2, direction="vertical")
                            break
                else:
                    continue
                break

        # return ARCGraph(non_background_vertical_connected_components_graph, self.name, self, "nbvcg")
        self.abstracted_graph = non_background_vertical_connected_components_graph
        self.abstraction = "nbvcg3"

    def get_non_background_horizontal_connected_components_graph(self, graph=None):
        if not graph:
            graph = self.get_2d_grid_graph()

        non_background_horizontal_connected_components_graph = nx.Graph()

        # for color in self.colors_included:
        node_id = 1
        for color in range(10):
            if color == 0:
                continue
            color_connected_components = []
            for row in range(self.height):
                color_nodes = (node for node, data in graph.nodes(data=True) if node[0] == row and data.get("color") == color)
                color_subgraph = graph.subgraph(color_nodes)
                color_connected_components.extend(list(connected_components(color_subgraph)))
            for i, component in enumerate(color_connected_components):
                non_background_horizontal_connected_components_graph.add_node(node_id, coordinates=sorted(list(component)), color=color, size=len(list(component)))
                node_id += 1


        # for node_1, attributes_1 in color_connected_components_graph.nodes(data=True):
        #     for node_2, attributes_2 in color_connected_components_graph.nodes(data=True):
        for node_1, node_2 in combinations(non_background_horizontal_connected_components_graph.nodes, 2):
            nodes_1 = non_background_horizontal_connected_components_graph.nodes[node_1]["coordinates"]
            nodes_2 = non_background_horizontal_connected_components_graph.nodes[node_2]["coordinates"]
            for n1 in nodes_1:
                for n2 in nodes_2:
                    if n1[0] == n2[0]:  # two nodes on the same row
                        for column_index in range(min(n1[1], n2[1])+1, max(n1[1], n2[1])):
                            if graph.nodes[n1[0], column_index]["color"] != self.background_color:
                                break
                        else:
                            non_background_horizontal_connected_components_graph.add_edge(node_1, node_2, direction="horizontal")
                            break
                    elif n1[1] == n2[1]:  # two nodes on the same column:
                        for row_index in range(min(n1[0], n2[0])+1, max(n1[0], n2[0])):
                            if graph.nodes[row_index, n1[1]]["color"] != self.background_color:
                                break
                        else:
                            non_background_horizontal_connected_components_graph.add_edge(node_1, node_2, direction="vertical")
                            break
                else:
                    continue
                break

        # return ARCGraph(non_background_vertical_connected_components_graph, self.name, self, "nbvcg")
        self.abstracted_graph = non_background_horizontal_connected_components_graph
        self.abstraction = "nbhcg"

    def get_non_background_horizontal_connected_components_graph_2(self, graph=None):
        if not graph:
            graph = self.get_2d_grid_graph()

        non_background_horizontal_connected_components_graph = nx.Graph()

        # for color in self.colors_included:
        node_id = 1
        for color in range(10):
            color_connected_components = []
            for row in range(self.height):
                color_nodes = (node for node, data in graph.nodes(data=True) if node[0] == row and data.get("color") == color)
                color_subgraph = graph.subgraph(color_nodes)
                color_connected_components.extend(list(connected_components(color_subgraph)))
            for i, component in enumerate(color_connected_components):
                if color != self.background_color:
                    non_background_horizontal_connected_components_graph.add_node(node_id, coordinates=sorted(list(component)), color=color, size=len(list(component)))
                    node_id += 1
                else:
                    component = list(component)
                    for node in component:
                        # if the node touches left or right edge of the image it is not included
                        if node[1] == 0 or node[1] == self.width - 1:
                            break
                    else:
                        non_background_horizontal_connected_components_graph.add_node(node_id, coordinates=component, color=color, size=len(component))
                        node_id += 1


        # for node_1, attributes_1 in color_connected_components_graph.nodes(data=True):
        #     for node_2, attributes_2 in color_connected_components_graph.nodes(data=True):
        for node_1, node_2 in combinations(non_background_horizontal_connected_components_graph.nodes, 2):
            nodes_1 = non_background_horizontal_connected_components_graph.nodes[node_1]["coordinates"]
            nodes_2 = non_background_horizontal_connected_components_graph.nodes[node_2]["coordinates"]
            for n1 in nodes_1:
                for n2 in nodes_2:
                    if n1[0] == n2[0]:  # two nodes on the same row
                        for column_index in range(min(n1[1], n2[1])+1, max(n1[1], n2[1])):
                            if graph.nodes[n1[0], column_index]["color"] != self.background_color:
                                break
                        else:
                            non_background_horizontal_connected_components_graph.add_edge(node_1, node_2, direction="horizontal")
                            break
                    elif n1[1] == n2[1]:  # two nodes on the same column:
                        for row_index in range(min(n1[0], n2[0])+1, max(n1[0], n2[0])):
                            if graph.nodes[row_index, n1[1]]["color"] != self.background_color:
                                break
                        else:
                            non_background_horizontal_connected_components_graph.add_edge(node_1, node_2, direction="vertical")
                            break
                else:
                    continue
                break

        # return ARCGraph(non_background_vertical_connected_components_graph, self.name, self, "nbvcg")
        self.abstracted_graph = non_background_horizontal_connected_components_graph
        self.abstraction = "nbhcg2"

    def get_non_background_horizontal_components_graph(self, graph=None):
        if not graph:
            graph = self.get_2d_grid_graph()

        non_background_horizontal_connected_components_graph = nx.Graph()

        # for color in self.colors_included:
        node_id = 1
        for row in range(self.height):
            # row_connected_components = []
            for color in range(10):
                if color == 0:
                    continue
                color_nodes = (node for node, data in graph.nodes(data=True) if node[0] == row and data.get("color") == color)
                color_subgraph = graph.subgraph(color_nodes)
                # color_connected_components.extend(list(connected_components(color_subgraph)))
                # row_connected_components.append(list(color_subgraph.nodes()))
            # for i, component in enumerate(row_connected_components):
            #     if not component:
            #         continue
                component = list(color_subgraph.nodes())
                if not component:
                    continue
                non_background_horizontal_connected_components_graph.add_node(node_id, coordinates=sorted(component), color=color, size=len(list(component)))
                node_id += 1


        # for node_1, attributes_1 in color_connected_components_graph.nodes(data=True):
        #     for node_2, attributes_2 in color_connected_components_graph.nodes(data=True):
        for node_1, node_2 in combinations(non_background_horizontal_connected_components_graph.nodes, 2):
            nodes_1 = non_background_horizontal_connected_components_graph.nodes[node_1]["coordinates"]
            nodes_2 = non_background_horizontal_connected_components_graph.nodes[node_2]["coordinates"]
            for n1 in nodes_1:
                for n2 in nodes_2:
                    if n1[0] == n2[0]:  # two nodes on the same row
                        for column_index in range(min(n1[1], n2[1])+1, max(n1[1], n2[1])):
                            if graph.nodes[n1[0], column_index]["color"] != self.background_color:
                                break
                        else:
                            non_background_horizontal_connected_components_graph.add_edge(node_1, node_2, direction="horizontal")
                            break
                    elif n1[1] == n2[1]:  # two nodes on the same column:
                        for row_index in range(min(n1[0], n2[0])+1, max(n1[0], n2[0])):
                            if graph.nodes[row_index, n1[1]]["color"] != self.background_color:
                                break
                        else:
                            non_background_horizontal_connected_components_graph.add_edge(node_1, node_2, direction="vertical")
                            break
                else:
                    continue
                break

        # return ARCGraph(non_background_vertical_connected_components_graph, self.name, self, "nbvcg")
        self.abstracted_graph = non_background_horizontal_connected_components_graph
        self.abstraction = "nbhcg3"

    def get_largest_rectangle_graph(self, graph=None):
        if not graph:
            # graph = self.graph
            graph = self.get_2d_grid_graph()

        # https://www.drdobbs.com/database/the-maximal-rectangle-problem/184410529?pgno=1
        def area(llx, lly, urx, ury):
            if llx > urx or lly > ury or [llx, lly, urx, ury] == [0, 0, 0, 0]:
                return 0
            else:
                return (urx - llx + 1) * (ury - lly + 1)

        def all_nb(llx, lly, urx, ury, g):
            for x in range(llx, urx+1):
                for y in range(lly, ury+1):
                    if (y, x) not in g:
                        return False
            return True

        lrg = nx.Graph()

        # for color in self.colors_included:
        node_id = 1
        for color in range(10):
            if color == 0:
                continue
            color_nodes = (node for node, data in graph.nodes(data=True) if data.get("color") == color)
            color_subgraph = graph.subgraph(color_nodes)
            subgraph_nodes = set(color_subgraph.nodes())
            i = 0
            while len(subgraph_nodes) != 0:
                best = [0, 0, 0, 0]
                for llx in range(self.width):
                    for lly in range(self.height):
                        for urx in range(self.width):
                            for ury in range(self.height):
                                cords = [llx, lly, urx, ury]
                                if area(*cords) > area(*best) and all_nb(*cords, subgraph_nodes):
                                    best = cords
                component = []
                for x in range(best[0], best[2] + 1):
                    for y in range(best[1], best[3] + 1):
                        component.append((y, x))
                        subgraph_nodes.remove((y, x))
                lrg.add_node(node_id, coordinates=component, color=color, size=len(component))
                node_id += 1
                i += 1

        # for node_1, attributes_1 in color_connected_components_graph.nodes(data=True):
        #     for node_2, attributes_2 in color_connected_components_graph.nodes(data=True):
        for node_1, node_2 in combinations(lrg.nodes, 2):
            nodes_1 = lrg.nodes[node_1]["coordinates"]
            nodes_2 = lrg.nodes[node_2]["coordinates"]
            for n1 in nodes_1:
                for n2 in nodes_2:
                    if n1[0] == n2[0]:  # two nodes on the same row
                        for column_index in range(min(n1[1], n2[1])+1, max(n1[1], n2[1])):
                            if graph.nodes[n1[0], column_index]["color"] != 0:
                                break
                        else:
                            lrg.add_edge(node_1, node_2, direction="horizontal")
                            break
                    elif n1[1] == n2[1]:  # two nodes on the same column:
                        for row_index in range(min(n1[0], n2[0])+1, max(n1[0], n2[0])):
                            if graph.nodes[row_index, n1[1]]["color"] != 0:
                                break
                        else:
                            lrg.add_edge(node_1, node_2, direction="vertical")
                            break
                else:
                    continue
                break

        # return ARCGraph(lrg, self.name, self, "lrg")
        self.abstracted_graph = lrg
        self.abstraction = "lrg"

    def get_connected_components_graph(self, graph=None):
        if not graph:
            graph = self.get_2d_grid_graph()

        color_connected_components_graph = nx.Graph()

        # for color in self.colors_included:
        node_id = 1
        for color in range(10):
            color_nodes = (node for node, data in graph.nodes(data=True) if data.get("color") == color)
            color_subgraph = graph.subgraph(color_nodes)
            color_connected_components = connected_components(color_subgraph)
            for i, component in enumerate(color_connected_components):
                color_connected_components_graph.add_node(node_id, coordinates=sorted(list(component)), color=color, size=len(list(component)))
                node_id += 1

        # for node_1, attributes_1 in color_connected_components_graph.nodes(data=True):
        #     for node_2, attributes_2 in color_connected_components_graph.nodes(data=True):
        for node_1, node_2 in combinations(color_connected_components_graph.nodes, 2):
            nodes_1 = color_connected_components_graph.nodes[node_1]["coordinates"]
            nodes_2 = color_connected_components_graph.nodes[node_2]["coordinates"]
            for n1 in nodes_1:
                for n2 in nodes_2:
                    if n1[0] == n2[0]:  # two nodes on the same row
                        for column_index in range(min(n1[1], n2[1]) + 1, max(n1[1], n2[1])):
                            if graph.nodes[n1[0], column_index]["color"] != self.background_color:
                                break
                        else:
                            color_connected_components_graph.add_edge(node_1, node_2, direction="horizontal")
                            break
                    elif n1[1] == n2[1]:  # two nodes on the same column:
                        for row_index in range(min(n1[0], n2[0]) + 1, max(n1[0], n2[0])):
                            if graph.nodes[row_index, n1[1]]["color"] != self.background_color:
                                break
                        else:
                            color_connected_components_graph.add_edge(node_1, node_2, direction="vertical")
                            break
                else:
                    continue
                break

        self.abstracted_graph = color_connected_components_graph
        self.abstraction = "ccg"

    def get_connected_components_graph_background_removed_2(self, graph=None):
        """
        different definition of 'background', also include background color along the edges of the frame
        :param graph:
        :return:
        """
        if not graph:
            # graph = self.graph
            graph = self.get_2d_grid_graph()
        ccgbr2 = nx.Graph()

        # for color in self.colors_included:
        node_id = 1
        for color in range(10):
            color_nodes = (node for node, data in graph.nodes(data=True) if data.get("color") == color)
            color_subgraph = graph.subgraph(color_nodes)
            color_connected_components = connected_components(color_subgraph)

            for i, component in enumerate(color_connected_components):
                if color != self.background_color:
                    ccgbr2.add_node(node_id, coordinates=sorted(list(component)), color=color, size=len(list(component)))
                    node_id += 1
                else:
                    component = list(component)
                    for node in component:
                        # if the node touches any edge of image it is not included
                        if node[0] == 0 or node[0] == self.height - 1 or node[1] == 0 or node[1] == self.width - 1:
                            break
                    else:
                        ccgbr2.add_node(node_id, coordinates=component, color=color, size=len(component))
                        node_id += 1

        for node_1, node_2 in combinations(ccgbr2.nodes, 2):
            nodes_1 = ccgbr2.nodes[node_1]["coordinates"]
            nodes_2 = ccgbr2.nodes[node_2]["coordinates"]
            for n1 in nodes_1:
                for n2 in nodes_2:
                    if n1[0] == n2[0]:  # two nodes on the same row
                        for column_index in range(min(n1[1], n2[1]) + 1, max(n1[1], n2[1])):
                            if graph.nodes[n1[0], column_index]["color"] != self.background_color:
                                break
                        else:
                            ccgbr2.add_edge(node_1, node_2, direction="horizontal")
                            break
                    elif n1[1] == n2[1]:  # two nodes on the same column:
                        for row_index in range(min(n1[0], n2[0]) + 1, max(n1[0], n2[0])):
                            if graph.nodes[row_index, n1[1]]["color"] != self.background_color:
                                break
                        else:
                            ccgbr2.add_edge(node_1, node_2, direction="vertical")
                            break
                else:
                    continue
                break

        # return ARCGraph(ccgbr2, self.name, self, "ccgbr")
        self.abstracted_graph = ccgbr2
        self.abstraction = "ccgbr2"

    def get_multicolor_connected_components_graph(self, graph=None):
        if not graph:
            graph = self.get_2d_grid_graph()
        multicolor_connected_components_graph = nx.Graph()

        non_background_nodes = [node for node, data in graph.nodes(data=True) if data["color"] != self.background_color]
        color_subgraph = graph.subgraph(non_background_nodes)
        multicolor_connected_components = connected_components(color_subgraph)

        node_id = 1
        for i, component in enumerate(multicolor_connected_components):
            sub_nodes = []
            sub_nodes_color = []
            for node in component:
                sub_nodes.append(node)
                sub_nodes_color.append(graph.nodes[node]["color"])
            combined = list(zip(sub_nodes, sub_nodes_color))
            sorted_combined = sorted(combined, key=lambda x: x[0])
            sub_nodes = [x[0] for x in sorted_combined]
            sub_nodes_color = [x[1] for x in sorted_combined]

            multicolor_connected_components_graph.add_node(node_id, coordinates=sub_nodes, color=sub_nodes_color,size=len(sub_nodes))
            node_id += 1

        # add edges between the abstracted nodes
        for node_1, node_2 in combinations(multicolor_connected_components_graph.nodes, 2):
            nodes_1 = multicolor_connected_components_graph.nodes[node_1]["coordinates"]
            nodes_2 = multicolor_connected_components_graph.nodes[node_2]["coordinates"]
            for n1 in nodes_1:
                for n2 in nodes_2:
                    if n1[0] == n2[0]:  # two nodes on the same row
                        for column_index in range(min(n1[1], n2[1]) + 1, max(n1[1], n2[1])):
                            if graph.nodes[n1[0], column_index]["color"] != self.background_color:
                                break
                        else:
                            multicolor_connected_components_graph.add_edge(node_1, node_2, direction="horizontal")
                            break
                    elif n1[1] == n2[1]:  # two nodes on the same column:
                        for row_index in range(min(n1[0], n2[0]) + 1, max(n1[0], n2[0])):
                            if graph.nodes[row_index, n1[1]]["color"] != self.background_color:
                                break
                        else:
                            multicolor_connected_components_graph.add_edge(node_1, node_2, direction="vertical")
                            break
                else:
                    continue
                break

        # return ARCGraph(multicolor_connected_components_graph, self.name, self, "mcccg")
        self.abstracted_graph = multicolor_connected_components_graph
        self.abstraction = "mcccg"

    def get_multicolor_components_with_diagonals_graph(self, graph=None):

        if not graph:
            graph = self.get_2d_grid_graph()

        with_diagonals = graph.copy()
        with_diagonals.add_edges_from([((x, y), (x + 1, y + 1)) for x in range(self.height-1) for y in range(self.width-1)]
                                      + [((x + 1, y), (x, y + 1)) for x in range(self.height-1) for y in range(self.width-1)])

        multicolor_connected_components_graph_with_diagnals = nx.Graph()

        non_background_nodes = [node for node, data in with_diagonals.nodes(data=True) if data["color"] != self.background_color]
        color_subgraph = with_diagonals.subgraph(non_background_nodes)
        multicolor_connected_components = connected_components(color_subgraph)

        node_id = 1
        for i, component in enumerate(multicolor_connected_components):
            sub_nodes = []
            sub_nodes_color = []
            for node in component:
                sub_nodes.append(node)
                sub_nodes_color.append(graph.nodes[node]["color"])
            multicolor_connected_components_graph_with_diagnals.add_node(node_id, coordinates=sub_nodes, color=sub_nodes_color,
                                                           size=len(sub_nodes))
            node_id += 1

        # for node_1, attributes_1 in color_connected_components_graph.nodes(data=True):
        #     for node_2, attributes_2 in color_connected_components_graph.nodes(data=True):
        for node_1, node_2 in combinations(multicolor_connected_components_graph_with_diagnals.nodes, 2):
            nodes_1 = multicolor_connected_components_graph_with_diagnals.nodes[node_1]["coordinates"]
            nodes_2 = multicolor_connected_components_graph_with_diagnals.nodes[node_2]["coordinates"]
            for n1 in nodes_1:
                for n2 in nodes_2:
                    if n1[0] == n2[0]:  # two nodes on the same row
                        for column_index in range(min(n1[1], n2[1])+1, max(n1[1], n2[1])):
                            if with_diagonals.nodes[n1[0], column_index]["color"] != 0:
                                break
                        else:
                            multicolor_connected_components_graph_with_diagnals.add_edge(node_1, node_2, direction="horizontal")
                            break
                    elif n1[1] == n2[1]:  # two nodes on the same column:
                        for row_index in range(min(n1[0], n2[0])+1, max(n1[0], n2[0])):
                            if with_diagonals.nodes[row_index, n1[1]]["color"] != 0:
                                break
                        else:
                            multicolor_connected_components_graph_with_diagnals.add_edge(node_1, node_2, direction="vertical")
                            break
                else:
                    continue
                break

        self.abstraction = "mcccg_d"
        self.abstracted_graph = multicolor_connected_components_graph_with_diagnals

    def get_no_abstraction_graph(self, graph=None):
        if not graph:
            graph = self.get_2d_grid_graph()

        no_abs_graph = nx.Graph()
        sub_nodes = []
        sub_nodes_color = []
        for node, data in graph.nodes(data=True):
            sub_nodes.append(node)
            sub_nodes_color.append(graph.nodes[node]["color"])
        no_abs_graph.add_node(1, coordinates=sub_nodes, color=sub_nodes_color, size=len(sub_nodes))

        # return ARCGraph(no_abs_graph, self.name, self, "na")
        self.abstracted_graph = no_abs_graph
        self.abstraction = "na"

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
