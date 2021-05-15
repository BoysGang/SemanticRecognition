import os
import networkx as nx
import matplotlib.pyplot as plt
import joblib


def read_graph_from_dict(graph_dict_path):
    G = nx.Graph()
    
    for line in open(graph_dict_path, 'r', encoding="utf8"):
        line = line.rstrip()
        line = line.split("\t")
        line[2] = float(line[2].replace(',', '.'))
        
        if (line[2] == 0.6):
            G.add_node(line[0])
            G.add_node(line[1])
            G.add_edge(line[0], line[1])

    return G

def is_noun(word):
    return word[-2:] != "ть" and word[-1] != "й"


def optimaze_graph(graph_dict_path, labels_list):
    # read base graph
    G = read_graph_from_dict(graph_dict_path)    

    # get required subgraph
    final_graph = nx.Graph()

    labels_list = [x.lower() for x in labels_list]
    labels_list = [x for x in labels_list if G.has_node(x)]

    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            for path in nx.all_simple_paths(G, labels_list[i], labels_list[j], cutoff=3):
                path = list(filter(is_noun, path))
                
                for k in range(len(path) - 1):
                    final_graph.add_node(path[k])
                    final_graph.add_node(path[k + 1])
                    final_graph.add_edge(path[k], path[k + 1])

    # save graph picture
    options = {
        'node_color': 'red',
        'node_size': 100,
        'with_labels': True,
        'font_color': 'black',
        'style': 'dotted',
    }

    nx.draw_circular(final_graph, **options)
    plt.savefig("mygraph.png")

    # save graph
    joblib.dump(final_graph, os.path.join(os.path.dirname(graph_dict_path), "graph.pkl"))

    
def sugest_classifiers(graph_dict_path, labels_list):
    # read base graph
    G = read_graph_from_dict(graph_dict_path)

    labels_list = [x.lower() for x in labels_list]
    labels_list = [x for x in labels_list if G.has_node(x)]

    suggested = []

    for label in labels_list:
        [suggested.append(n) for n in G.neighbors(label) if is_noun(n)]

    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            for path in nx.all_simple_paths(G, labels_list[i], labels_list[j], cutoff=3):
                path = list(filter(is_noun, path))
                
                suggested.extend(path)

    print("Related semantic words:")
    print(set(suggested))
    
if __name__ == '__main__':
    #optimaze_graph("graph/dict__base.txt", ["Автомобиль", "Велосипед", "Собака", "Человек", "Мотоцикл", "Колесо"])
    sugest_classifiers("graph/dict__base.txt", ["Автомобиль", "Велосипед", "Собака", "Человек", "Мотоцикл", "Колесо"])