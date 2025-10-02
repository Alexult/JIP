import networkx as nx
from pathlib import Path
from matplotlib import pyplot as plt


class Grid:
    """Class that represents the physical grid and models the electricity flow"""

    def __init__(self, graph):
        self.graph = graph

    def update_capacity(self, trades: list) -> None:
        """Update the capacity of the grid based on the trades"""

        for trade in trades:
            if trade[2] == 0:
                continue

            path = nx.shortest_path(self.graph, trade[0], 'Transformer')
            path_edges = list(zip(path, path[1:]))

            for edge in path_edges:
                if edge[1] == "Transformer":
                    new_flow = self.graph.nodes["Transformer"]['flow'] - trade[2]
                    if abs(new_flow) > self.graph.nodes["Transformer"]['capacity']:
                        raise Exception("New flow exceeds transformer capacity")
                    self.graph.nodes["Transformer"]['flow'] = new_flow
                    continue

                new_flow = self.graph[edge[0]][edge[1]]['flow'] - trade[2]
                if abs(new_flow) > self.graph[edge[0]][edge[1]]['capacity']:
                    raise Exception("New flow exceeds capacity")
                self.graph[edge[0]][edge[1]]['flow'] = new_flow

            path = nx.shortest_path(self.graph, 'Transformer', trade[1])
            path_edges = list(zip(path, path[1:]))

            for edge in path_edges:
                if edge[0] == 'Transformer':
                    new_flow = self.graph.nodes["Transformer"]['flow'] + trade[2]
                    if abs(new_flow) > self.graph.nodes["Transformer"]['capacity']:
                        raise Exception("New flow exceeds transformer capacity")
                    self.graph.nodes["Transformer"]['flow'] = new_flow
                    continue

                new_flow = self.graph[edge[0]][edge[1]]['flow'] + trade[2]
                if abs(new_flow) > self.graph[edge[0]][edge[1]]['capacity']:
                    raise Exception("New flow exceeds capacity")
                self.graph[edge[0]][edge[1]]['flow'] = new_flow

    def retrieve_capacity(self, n: str, u: str) -> int:
        """Finds the maximum capacity trade from node u to u"""

        path = nx.shortest_path(self.graph, n, 'Transformer')
        path_edges = list(zip(path, path[1:]))

        flow = float('inf')
        if n == 'Transformer' or u == 'Transformer':
            flow = self.graph.nodes["Transformer"]['capacity'] - self.graph.nodes["Transformer"]['flow']

        for edge in path_edges:
            if edge[0] == "Transformer" or edge[1] == 'Transformer':
                continue

            cap = self.graph[edge[0]][edge[1]]['capacity'] + self.graph[edge[0]][edge[1]]['flow']
            if cap < flow:
                flow = cap

        path = nx.shortest_path(self.graph, 'Transformer', u)
        path_edges = list(zip(path, path[1:]))

        for edge in path_edges:
            if edge[0] == "Transformer" or edge[1] == 'Transformer':
                continue

            cap = self.graph[edge[0]][edge[1]]['capacity'] - self.graph[edge[0]][edge[1]]['flow']
            if cap < flow:
                flow = cap

        return flow


def read_grid(read_file: str) -> Grid:
    """
    Reads the grid layout including capacities from the file specified
    File must be in the Data directory and follow specific formatting:
    Nodes:
    [("Transformer";cap),("name"), ()]

    Edges:
    [("Start";"end";cap)]
    """
    data_folder = Path(__file__).parent.parent
    file = data_folder / "data" / read_file

    with open(file) as f:
        text = f.read()
        text = text.replace(" ", "").split("\n")
        text = [x for x in text if x != ""]
        if text[0].lower() != "nodes:":
            raise Exception("file needs to start with 'nodes:'")
        if text[2].lower() != "edges:":
            raise Exception("file needs to start with 'edges:'")
        if text[1][0] != "[" and text[1][-1] != "]":
            raise Exception("nodes need to be surrounded by '[]'")
        if text[3][0] != "[" and text[3][-1] != "]":
            raise Exception("edges need to be surrounded by '[]'")

        nodes = text[1][1:-1].split(",")
        edges = text[3][1:-1].split(",")
        transformer = nodes[0][1:-1].split(';')
        if transformer[0] != "Transformer" or not transformer[1].isdigit():
            raise Exception("Incorrect transformer format")
        nodes = nodes[1:]

        def transform(x):
            text = x[1:-1].split(";")
            if not text[2].isdigit() and text[2] != '-1':
                raise Exception("capacity needs to be an integer")
            return text[0], text[1], {'capacity': int(text[2]), 'flow': 0}

        edges = list(map(transform, edges))
        g = nx.Graph()
        g.add_node("Transformer", capacity=int(transformer[1]), flow=0)
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)

        return Grid(g)


# dummy code that reads the grid, adds some trades, and plots the grid
# grid = read_grid("graph.txt")
#
# grid.update_capacity([("Transformer", '4', 2)])
#
# print(grid.retrieve_capacity('Transformer', '1'))
#
# pos = nx.spring_layout(grid.graph)
# nx.draw_networkx_nodes(grid.graph, pos)
# nx.draw_networkx_edges(grid.graph, pos)
# edge_labels = nx.get_edge_attributes(grid.graph, "capacity")
# nx.draw_networkx_edge_labels(grid.graph, pos, edge_labels)
# nx.draw_networkx_labels(grid.graph, pos)
# plt.show()
#
# pos = nx.spring_layout(grid.graph)
# nx.draw_networkx_nodes(grid.graph, pos)
# nx.draw_networkx_edges(grid.graph, pos)
# edge_labels = nx.get_edge_attributes(grid.graph, "flow")
# nx.draw_networkx_edge_labels(grid.graph, pos, edge_labels)
# nx.draw_networkx_labels(grid.graph, pos)
# plt.show()
