import networkx as nx
from networkx.algorithms.centrality import closeness
from SemanticGraph import SemanticGraph


class SemanticCorrection:
    def __init__(self, labels, predictions, semantic_graph: SemanticGraph):
        self.__predictions = predictions
        self.__labels = [label.lower() for label in labels]
        self.__graph = semantic_graph
        self.__nodes = list(self.__graph.nodes)

    def execute(self, d=0.6, threshold=1, weight=1):
        nodes = self.__nodes

        closenesses = list()
        normalization_closenesses = list()

        for label, prediction in zip(self.__labels, self.__predictions):
            node_index = nodes.index(label)

            prediction = 1 - ((prediction - min(self.__predictions)) / (max(self.__predictions) - min(self.__predictions)) * 0.5)

            closeness = self.__modified_dijkstra(d, threshold, node_index, prediction)
            closenesses.append(closeness)

            closeness = self.__modified_dijkstra(d, threshold, node_index)
            normalization_closenesses.append(closeness)

            # closenesses.append(self.__compute_closeness1(label, prediction, d, weight, threshold))
            # normalization_closenesses.append(self.__compute_closeness1(label, 1, d, weight, threshold))

        classes_width = [sum(closeness) 
            for closeness in closenesses]

        normalization_classes_width = [sum(closeness) 
            for closeness in normalization_closenesses]

        normalized_width = [classes_width[i] / normalization_classes_width[i] 
            for i in range(len(classes_width))]

        clusters = self.__concept_union(classes_width, normalized_width, closenesses)

        return clusters, normalized_width

    # def __compute_closeness1(self, label, prediction, d, weight, threshold):
    #     closeness = list()

    #     for lbl in self.__labels:
    #         path = nx.shortest_path(self.__graph, source=label, target=lbl, weight=weight)
    #         edge_number = len(path) - 1

    #         close = 1
    #         for i in range(edge_number):    
    #             if close >= threshold:
    #                 close *= (d * weight)
    #             else:
    #                 close = 0
    #                 break

    #         closeness.append(close)

    #     return closeness

    def __concept_union(self, classes_width, normalized_width, closenesses):
        clusters = self.__labels[::]

        while True:
            was_unioned = False

            for i in range(len(clusters) - 1):
                for j in range(i + 1, len(clusters)):
                    numerator = 0
                    denominator = 0

                    for closeness_i, closeness_j in zip(closenesses[i], closenesses[j]):
                        if closeness_i != 0 and closeness_j != 0:
                            numerator += (closeness_i + closeness_j + (closeness_i * closeness_j)**(1/3))

                    denominator = classes_width[i] + classes_width[j]

                    if numerator / denominator > 1.2:
                        cluster = clusters[i] + ', ' + clusters[j]

                        cluster_closeness = [max(closeness_i, closeness_j) 
                            for closeness_i, closeness_j in zip(closenesses[i], closenesses[j])]

                        max_width = max(normalized_width[i], normalized_width[j])
                        min_width = min(normalized_width[i], normalized_width[j])
                        cluster_normalized_width = max_width * ((1 + max_width * min_width**2) / (max_width**3 + min_width**3))

                        clusters.pop(j)
                        clusters.pop(i)
                        clusters.append(cluster)

                        closenesses.pop(j)
                        closenesses.pop(i)
                        closenesses.append(cluster_closeness)

                        normalized_width.pop(j)
                        normalized_width.pop(i)
                        normalized_width.append(cluster_normalized_width)

                        classes_width.pop(j)
                        classes_width.pop(i)
                        classes_width.append(sum(cluster_closeness))

                        was_unioned = True

                        break

                if was_unioned:
                    break

            if not was_unioned:
                break

        return clusters

    def __modified_dijkstra(self, d, threshold, start_node=0, prediction=1):
        nodes_number = self.__graph.number_of_nodes()
        visited = [False] * nodes_number
        closeness = [float('inf')] * nodes_number
        closeness[start_node] = prediction

        nodes = list(self.__graph.nodes)

        for _ in range(nodes_number):
 
            # Pick the minimum distance vertex from
            # the set of vertices not yet processed.
            # u is always equal to src in first iteration
            u = self.__minDistance(closeness, visited)
 
            # Put the minimum distance vertex in the
            # shotest path tree
            visited[u] = True

            if closeness[u] < threshold:
                continue

            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the shotest path tree
            for vertex in self.__graph.neighbors(nodes[u]):
                computed_closeness = self.__compute_closeness(closeness[u], d, weight=0.6)
                v = nodes.index(vertex)

                if computed_closeness < closeness[v] and not visited[v]:
                    closeness[v] = computed_closeness

        closeness = [0 if c == float('inf') else c for c in closeness]

        return closeness

    def __compute_closeness(self, current_closeness, d, weight=1):
        return current_closeness * d * weight

    def __minDistance(self, closeness, visited):
 
        # Initilaize minimum distance for next node
        min = float('inf')
        min_index = 0

        # Search not nearest vertex not in the
        # shortest path tree
        for v in range(self.__graph.number_of_nodes()):
            if closeness[v] < min and visited[v] == False:
                min = closeness[v]
                min_index = v
 
        return min_index