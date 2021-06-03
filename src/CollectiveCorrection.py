import networkx as nx

from CorrectionMethod import CorrectionMethod
from SemanticGraph import SemanticGraph


class CollectiveCorrection(CorrectionMethod):
    def __init__(self, radius=2, threshold=0.2):
        self.__radius = radius
        self.__threshold = threshold

    def execute(self, labels, predictions, semantic_graph: SemanticGraph):
        merged_concepts = [False] * len(labels)
        clusters = list()
        merged_clusters = list()

        # do not consider concepts with low probability
        for i in range(len(labels)):
            if predictions[i] <= self.__threshold:
                merged_clusters.append([labels[i]])
                merged_concepts[i] = True

        # compute cluster for each concept
        for i in range(len(labels)):
            if not merged_concepts[labels.index(labels[i])]:
                cluster = []

                for j in range(len(labels)):
                    path_length = nx.shortest_path_length(semantic_graph, labels[i], labels[j])
                    
                    if path_length <= self.__radius:
                        cluster.append(labels[j])

                clusters.append(cluster)

        # merging common elem in cluster
        for i in range(len(clusters)):
            max_common_num = 0
            elems_to_merge = []

            for j in range(i + 1, len(clusters)):
                common_elems = list(set(clusters[i]).intersection(clusters[j]))
                common_elems = [el for el in common_elems if not merged_concepts[labels.index(el)]]

                length = len(common_elems)
                if length > max_common_num:
                    elems_to_merge = common_elems

            if elems_to_merge:
                for elem in elems_to_merge:
                    merged_concepts[labels.index(elem)] = True
                merged_clusters.append(elems_to_merge)
        
        # add not merged concepts as single concept
        for i in range(len(merged_concepts)):
            if not merged_concepts[i]:
                merged_clusters.append([labels[i]])

        # compute clusters prbabilities
        cluster_probs = list()
        for cluster in merged_clusters:
            prob_sum = 0

            for label in cluster:
                index = labels.index(label)
                prob_sum += predictions[index]

            cluster_probs.append(prob_sum)

        # normalization
        normalized_cluster_probs = self._normalize(cluster_probs)

        # new results labels
        cluster_labels = list()
        for cluster in merged_clusters:
            cluster_labels.append(', '.join(cluster))

        return cluster_labels, normalized_cluster_probs
