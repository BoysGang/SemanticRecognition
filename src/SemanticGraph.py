import os
import networkx as nx
import matplotlib.pyplot as plt
import joblib

class SemanticGraph(nx.Graph):
    def __init__(self):
        super().__init__()

    def read_from_dictionary(self, dict_path):
        for line in open(dict_path, 'r', encoding="utf8"):
            line = line.rstrip()
            line = line.split(" ")
            
            self.add_node(line[0])
            self.add_node(line[1])
            self.add_edge(line[0], line[1])

        return self

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, pickle_path):
        return joblib.load(pickle_path)

    def save_graph_image(self, path):
        # save graph picture
        options = {
            'node_color': 'red',
            'node_size': 100,
            'with_labels': True,
            'font_color': 'black',
            'style': 'dotted',
        }

        nx.draw_circular(self, **options)
        plt.savefig(path)