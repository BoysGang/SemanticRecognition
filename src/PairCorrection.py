import networkx as nx
from CorrectionMethod import CorrectionMethod
from SemanticGraph import SemanticGraph


class PairCorrection(CorrectionMethod):
    def __init__(self, radius=3, threshold=0.5):
        self.__r = radius
        self.__threshold = threshold

    def execute(self, labels, predictions, semantic_graph: SemanticGraph):
        clusters = list()

        looked_concepts = list()
        for i in range(len(labels)):
            if predictions[i] >= self.__threshold:
                looked_concepts.append(True)
            else:
                looked_concepts.append(False)
                clusters.append([labels[i]])

        for i in range(len(looked_concepts)):
            if looked_concepts[i]:
                cluster = [labels[i]]

                for j in range(i + 1, len(looked_concepts)):
                    if looked_concepts[j] and nx.shortest_path_length(semantic_graph, labels[i], labels[j]) <= self.__r:
                        cluster.append(labels[j])
                        looked_concepts[j] = False

                looked_concepts[i] = False

                clusters.append(cluster)

        cluster_probs = list()
        for cluster in clusters:
            prob_sum = 0

            for label in cluster:
                index = labels.index(label)
                prob_sum += predictions[index]

            cluster_probs.append(prob_sum)

        max_cluster_prob = max(cluster_probs)
        normalized_cluster_probs = list(map(lambda x: x / max_cluster_prob, cluster_probs))

        cluster_labels = list()
        for cluster in clusters:
            cluster_labels.append(', '.join(cluster))

        return cluster_labels, normalized_cluster_probs