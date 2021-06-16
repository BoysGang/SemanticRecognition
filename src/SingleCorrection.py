import networkx as nx

from CorrectionMethod import CorrectionMethod
from SemanticGraph import SemanticGraph


class SingleCorrection(CorrectionMethod):
    def __init__(self, radius=2, damping=0.8):
        self.__radius = radius
        self.__damping = damping

    def execute(self, labels, predictions, semantic_graph: SemanticGraph):
        corrected_predictions = [0] * len(predictions)
        
        # Increase each class probability by close classes probabilities
        for i in range(len(labels)):
            for j in range(len(labels)):
                path_length = nx.shortest_path_length(semantic_graph, labels[i], labels[j])
                    
                if path_length <= self.__radius:
                    corrected_predictions[i] += predictions[j] * self.__damping**path_length
            
        # Normalization
        corrected_predictions = self._normalize(corrected_predictions)

        return labels, corrected_predictions
