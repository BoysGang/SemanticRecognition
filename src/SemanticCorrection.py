from CorrectionMethod import CorrectionMethod
from SemanticGraph import SemanticGraph


class SemanticCorrection:
    def __init__(self, labels, predictions, semantic_graph: SemanticGraph):
        self.__predictions = predictions
        self.__labels = [label.lower() for label in labels]
        self.__graph = semantic_graph

    # Sets specific implementation of the semantic correction algorithm
    def set_method(self, method: CorrectionMethod):
        self.__method = method

    # Applying semantic correction algorithm
    def apply(self):
        return self.__method.execute(self.__labels, self.__predictions, self.__graph)
