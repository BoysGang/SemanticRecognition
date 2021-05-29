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

    def filter(self, classes, depth=3):
        # classes preproccesing
        classes = [x.lower() for x in classes]
        
        temp = []
        for cl in classes:
            if self.has_node(cl):
                temp.append(cl)
            else:
                print("base graph has no node:", cl)
        
        classes = temp

        # get required subgraph
        final_graph = SemanticGraph()

        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                for path in nx.all_simple_paths(self, classes[i], classes[j], cutoff=depth):
                    
                    for k in range(len(path) - 1):
                        final_graph.add_node(path[k])
                        final_graph.add_node(path[k + 1])
                        final_graph.add_edge(path[k], path[k + 1])

        self.clear()
        self.add_nodes_from(final_graph.nodes)
        self.add_edges_from(final_graph.edges)

    def save(self, path):
        joblib.dump(self, path + '.pkl')

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