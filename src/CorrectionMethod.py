from abc import ABC, abstractmethod

from SemanticGraph import SemanticGraph


class CorrectionMethod(ABC):
    @abstractmethod
    def execute(self, labels, predictions, semantic_graph: SemanticGraph):
        pass

    def _normalize(self, predictions):
        max_prob = max(predictions)
        predictions = list(map(lambda x: x / max_prob, predictions))

        return predictions