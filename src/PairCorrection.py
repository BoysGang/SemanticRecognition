import networkx as nx
from CorrectionMethod import CorrectionMethod
from SemanticGraph import SemanticGraph


class PairCorrection(CorrectionMethod):
    def __init__(self, radius=2, threshold=0.2):
        self.__radius = radius
        self.__threshold = threshold

    def execute(self, labels, predictions, semantic_graph: SemanticGraph):
        clusters = list()

        # mark low probabilities concept as viewed and add as single vertex clusters
        viewed_concepts = [True] * len(labels)
        for i in range(len(labels)):
            if predictions[i] < self.__threshold:
                viewed_concepts[i] = False
                clusters.append([labels[i]])

        # look for cluster merging
        for i in range(len(viewed_concepts)):
            if viewed_concepts[i]:
                cluster = [labels[i]]

                for j in range(i + 1, len(viewed_concepts)):
                    path_length = nx.shortest_path_length(semantic_graph, labels[i], labels[j])
                    
                    if viewed_concepts[j] and path_length <= self.__radius:
                        viewed_concepts[j] = False
                        cluster.append(labels[j])

                viewed_concepts[i] = False
                clusters.append(cluster)

        # compute clusters prbabilities
        cluster_probs = list()
        for cluster in clusters:
            prob_sum = 0

            for label in cluster:
                index = labels.index(label)
                prob_sum += predictions[index]

            cluster_probs.append(prob_sum)

        # normalization
        max_cluster_prob = max(cluster_probs)
        normalized_cluster_probs = list(map(lambda x: x / max_cluster_prob, cluster_probs))

        # new results labels
        cluster_labels = list()
        for cluster in clusters:
            cluster_labels.append(', '.join(cluster))

        return cluster_labels, normalized_cluster_probs