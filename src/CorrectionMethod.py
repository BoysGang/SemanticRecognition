from abc import ABC, abstractmethod

from SemanticGraph import SemanticGraph


class CorrectionMethod(ABC):
    @abstractmethod
    def execute(self, labels, predictions, semantic_graph: SemanticGraph):
        pass