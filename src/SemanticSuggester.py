import networkx as nx
import random

from SemanticGraph import SemanticGraph


class SemanticSuggester:
    @classmethod
    def suggest_by_shortest_paths(cls, base_graph, classes, print_paths=False):
        classes = cls.__classes_preproccesing(base_graph, classes)
        suggested = []

        if print_paths:
            print("Shortest paths between classes:")

        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                for path in nx.all_shortest_paths(base_graph, classes[i], classes[j]):
                    if print_paths:
                        print(path)
                    
                    path = list(filter(cls.__is_noun, path))
                    suggested.extend(path)

        suggested = set(suggested)
        for cl in classes:
            suggested.discard(cl)
        
        print("\nRelated semantic classes:")
        print(suggested)

    @classmethod
    def suggest_neighbors_on_depth(cls, base_graph, classes, depth, output_length=10):
        classes = cls.__classes_preproccesing(base_graph, classes)

        print("\nNeighbors on depth", str(depth) + ":")

        for i in range(len(classes)):
            neighbors = cls.__get_neighbors_on_depth(base_graph, classes[i], depth)
            neighbors = list(filter(cls.__is_noun, neighbors))
            neighbors = set(neighbors)
            neighbors = set(random.sample(neighbors, output_length)) 

            print()
            print(classes[i], ":", neighbors)
            

    @classmethod
    def __get_neighbors_on_depth(cls, graph, node, depth):
        path_lengths = nx.single_source_shortest_path_length(graph, node, depth)
        neighbors = [node for node, length in path_lengths.items()
                        if length == depth]
        return neighbors

    @classmethod
    def __is_noun(cls, word):
        return word[-2:] != "ть" and word[-1] != "й"

    @classmethod
    def __classes_preproccesing(cls, graph, classes):
        classes = [x.lower() for x in classes]
        classes = [x for x in classes if graph.has_node(x)]

        return classes
