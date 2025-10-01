import networkx as nx
from pathlib import Path

from matplotlib import pyplot as plt


class Grid:

    def __init__(self, graph):
        self.graph = graph


    def update_capacity(self, trades: list) -> None:
        for trade in trades:
            path = nx.shortest_path(self.graph, trade[0], trade[1])
            path_edges = list(zip(path, path[1:]))

            for edge in path_edges:
                new_flow = self.graph[edge[0]][edge[1]]['flow'] + trade[2]
                if new_flow > self.graph[edge[0]][edge[1]]['capacity']:
                    raise Exception("New flow exceeds capacity")
                self.graph[edge[0]][edge[1]]['flow'] = self.graph[edge[0]][edge[1]]['flow'] + trade[2]

    def retrieve_capacity(self, n: str, u: str) -> int:
        path = nx.shortest_path(self.graph, n, u)
        path_edges = list(zip(path, path[1:]))
        max = 0
        for edge in path_edges:
            cap = self.graph[edge[0]][edge[1]]['capacity'] - self.graph[edge[0]][edge[1]]['flow']
            if cap > max:
                max = cap
        return max


def read_grid(read_file: str) -> Grid:

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

        def transform(x):
            text = x[1:-1].split(";")
            if not text[2].isdigit():
                raise Exception("capacity needs to be an integer")
            return (text[0], text[1], {'capacity': int(text[2]), 'flow': 0})

        edges = list(map(transform, edges))
        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)

        return Grid(g)


# grid = read_grid("graph.txt")
#
# grid.update_capacity([('1', '3', 1), ('4', '2', 1)])
#
# print(grid.retrieve_capacity('1', '3'))
# print(grid.retrieve_capacity('4', '1'))
#
# pos = nx.spring_layout(grid.graph)
# nx.draw_networkx_nodes(grid.graph, pos)
# nx.draw_networkx_edges(grid.graph, pos)
# edge_labels = nx.get_edge_attributes(grid.graph, "flow")
# nx.draw_networkx_edge_labels(grid.graph, pos, edge_labels)
# nx.draw_networkx_labels(grid.graph, pos)
# plt.show()