# abstraction.py

import os
import json
import networkx as nx
from networkx.algorithms.components import connected_components
from itertools import combinations

class Image:
    """
    Use the abstraction on file for the task and return using the user specified encoding
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
        "nbvcg": "get_non_background_vertical_connected_components_graph",
        "na": "get_no_abstraction_graph",
        "lrg": "get_largest_rectangle_graph",
        "ccgbr": "get_connected_components_graph_background_removed",
        "ccgbr2": "get_connected_components_graph_background_removed_2",
        "nbhcg2": "get_non_background_horizontal_connected_components_graph_2",
        "ccg": "get_connected_components_graph",
        "mcccg": "get_multicolor_connected_components_graph"
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

    def get_graph_encoded_string(self, encoding="node_edge_set"):
        prompt = ""
        if encoding == "node_edge_set":
            node_list = ""
            edge_list = ""
            for node, attrs in self.abstracted_graph.nodes(data=True):
                attrs_str = ', '.join(f'{key}={value}' for key, value in attrs.items())
                node_str = f'Node {node}: {attrs_str}\n'
                node_list += node_str

            for node1, node2, attrs in self.abstracted_graph.edges(data=True):
                attrs_str = ', '.join(f'{key}={value}' for key, value in attrs.items())
                edge_str = f'Edge {node1},{node2}: {attrs_str}\n'
                edge_list += edge_str

            prompt = "Nodes:\n" + node_list + "\nEdges:\n" + edge_list + "\n"

        elif encoding == "graph_json":
            # Serialize full graph structure (nodes + edges) in JSON format
            prompt = json.dumps(nx.node_link_data(self.abstracted_graph))

        elif encoding == "object_json":
            # Serialize only node (object) data in JSON format
            nodes = nx.node_link_data(self.abstracted_graph)["nodes"]
            for node in nodes:
                node.pop("id")
            prompt = json.dumps(nodes)
            prompt = "Objects:\n" + prompt + "\n"

        elif encoding == "object_descriptor":
            # Human-readable format with object descriptors
            node_list = ""
            for node, attrs in self.abstracted_graph.nodes(data=True):
                attrs_str = ', '.join(f'{key}={value}' for key, value in attrs.items())
                node_str = f'Object {node}: {attrs_str}\n'
                node_list += node_str
            prompt = "Objects:\n" + node_list

        elif encoding == "object_json_w_edge":
            # Serialize node (object) data with neighboring nodes (edges) in JSON format
            graph_data = nx.node_link_data(self.abstracted_graph)
            nodes = graph_data["nodes"]
            for node in nodes:
                node_id = node['id']
                node['neighbors'] = [n for n in self.abstracted_graph.neighbors(node_id)]
            prompt = json.dumps(nodes)
            prompt = "Objects:\n" + prompt + "\n"

        elif encoding == "object_descriptor_w_edge":
            # Human-readable format with object descriptors and neighboring nodes (edges)
            node_list = ""
            for node, attrs in self.abstracted_graph.nodes(data=True):
                attrs_str = ', '.join(f'{key}={value}' for key, value in attrs.items())
                attrs_str += ', neighbors=[' + ', '.join(f"Object {n}" for n in self.abstracted_graph.neighbors(node)) + ']'
                node_str = f'Object {node}: {attrs_str}\n'
                node_list += node_str
            prompt = "Objects:\n" + node_list

        elif encoding == "object_json_words":
            # Object data with color represented as words
            convert = lambda i: self.digit_to_word[i]
            nodes = nx.node_link_data(self.abstracted_graph)["nodes"]
            for node in nodes:
                node.pop("id")
                if self.abstraction in ["na", "mcccg", "mcccg_d"]:
                    node["color"] = [convert(c) for c in node["color"]]
                else:
                    node["color"] = convert(node["color"])
            prompt = json.dumps(nodes)
            prompt = "Objects:\n" + prompt + "\n"

        return prompt

def get_abstraction_method(task_id):
    """
    Determines the abstraction method to use for a given task_id.
    """
    # Path to the solutions directory
    solutions_dir = "/Users/vitsiozo/Desktop/MSc AI/Modules/Project/ARC/object_encoding_scripts/ARGA-solutions/" 
    solutions_file = os.path.join(solutions_dir, f"solutions_{task_id}.json")

    # Check if the solutions file exists
    if os.path.exists(solutions_file):
        with open(solutions_file, 'r') as f:
            solutions_data = json.load(f)
        abstraction_method_name = solutions_data.get("abstraction", "nbccg")
        #print(f"Found abstraction method '{abstraction_method_name}' from solutions file for {task_id}.")
    else:
        abstraction_method_name = "nbccg"
        print(f"Solutions file not found for task {task_id}. Defaulting to abstraction method '{abstraction_method_name}'.")

    # Ensure the abstraction method exists
    if abstraction_method_name not in Image.abstraction_ops:
        print(f"Abstraction method '{abstraction_method_name}' not recognized. Defaulting to 'nbccg'.")
        abstraction_method_name = "nbccg"

    return abstraction_method_name

def generate_abstracted_task(task_data, task_id, encoding="object_json", test_input_index=None):
    """
    Generates the abstracted representation of a task using the appropriate abstraction method.
    
    Args:
        task_data: The task data containing 'train' and 'test' examples.
        task_id: The unique ID of the task being processed.
        encoding: The encoding method used (e.g., 'object_json').
        test_input_index: The index of the test input to generate the abstraction for. If None, process all test inputs.
    
    Returns:
        A string representing the abstracted task (training examples + one specific test input).
    """
    abstraction_method_name = get_abstraction_method(task_id)
    images = []

    # Process training examples (included in the prompt for every test input)
    for idx, pair in enumerate(task_data['train']):
        input_image = Image(grid=pair['input'], name=f'train_input_{idx+1}')
        output_image = Image(grid=pair['output'], name=f'train_output_{idx+1}')
        
        # Call the selected abstraction method on the training images
        abstraction_method = Image.abstraction_ops[abstraction_method_name]
        getattr(input_image, abstraction_method)()
        getattr(output_image, abstraction_method)()
        
        images.append((input_image, output_image))

    # Process test examples
    if test_input_index is not None:
        # Only process the specific test input based on the provided index
        pair = task_data['test'][test_input_index]
        input_image = Image(grid=pair['input'], name=f'test_input_{test_input_index+1}')
        getattr(input_image, abstraction_method)()
        output_image = None
        if 'output' in pair:
            output_image = Image(grid=pair['output'], name=f'test_output_{test_input_index+1}')
            getattr(output_image, abstraction_method)()
        images.append((input_image, output_image))
    else:
        # Process all test inputs if no specific index is provided (this can be used for debugging)
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
        input_encoding = input_image.get_graph_encoded_string(encoding=encoding)
        abstracted_task_str += f'{input_encoding}\n'

        if output_image:
            abstracted_task_str += f'Output Image ({output_image.name}):\n'
            abstracted_task_str += f'Image size: {output_image.image_size}\n'
            output_encoding = output_image.get_graph_encoded_string(encoding=encoding)
            abstracted_task_str += f'{output_encoding}\n'

    return abstracted_task_str

