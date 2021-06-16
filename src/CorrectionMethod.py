from abc import ABC, abstractmethod

from SemanticGraph import SemanticGraph


class CorrectionMethod(ABC):
    # Execute semantic correction algorithm
    @abstractmethod
    def execute(self, labels, predictions, semantic_graph: SemanticGraph):
        pass
    
    # Normalize results of semantic correction
    def _normalize(self, predictions):
        max_prob = max(predictions)
        predictions = list(map(lambda x: x / max_prob, predictions))

        return predictions
