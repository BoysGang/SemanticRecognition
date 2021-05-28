import networkx as nx

from SemanticGraph import SemanticGraph


class SemanticGraphFilter:
    @classmethod
    def filter(cls, base_graph, classes, depth=3):
        # get required subgraph
        final_graph = SemanticGraph()

        classes = cls.__classes_preproccesing(base_graph, classes)

        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                for path in nx.all_simple_paths(base_graph, classes[i], classes[j], cutoff=depth):
                    
                    for k in range(len(path) - 1):
                        final_graph.add_node(path[k])
                        final_graph.add_node(path[k + 1])
                        final_graph.add_edge(path[k], path[k + 1])

        return final_graph

    @classmethod
    def suggest_classifiers(cls, base_graph, classes, print_paths=False):
        classes = cls.__classes_preproccesing(base_graph, classes)
        suggested = []

        # for one class suggest neighbors only 
        if len(classes) == 1:
            [suggested.append(n) for n in base_graph.neighbors(classes[0]) if cls.__is_noun(n)]
        elif print_paths:
            print("Shortest paths between classes:")

        # for two or more classes suggest by shortest paths
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
    def __is_noun(cls, word):
        return word[-2:] != "ть" and word[-1] != "й"

    @classmethod
    def __classes_preproccesing(cls, graph, classes):
        classes = [x.lower() for x in classes]
        classes = [x for x in classes if graph.has_node(x)]

        return classes
