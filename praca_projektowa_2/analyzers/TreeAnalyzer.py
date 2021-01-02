from praca_projektowa_2.analyzers.Analyzer import Analyzer
from praca_projektowa_2.classifiers.TreeClassifier import TreeClassifier


class TreeAnalyzer(Analyzer):
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    classifier = None
    result_mapping_legend = {}

    def __init__(self):
        pass

    @classmethod
    def build(cls, x_train, x_test, y_train, y_test, result_mapping_legend):
        analyzer = TreeAnalyzer()
        analyzer.x_train = x_train
        analyzer.x_test = x_test
        analyzer.y_train = y_train
        analyzer.y_test = y_test
        analyzer.result_mapping_legend = result_mapping_legend
        return analyzer

    def findBestClassifier(self):
        classifier = TreeClassifier.build(self.x_train, self.x_test, self.y_train, self.y_test)
        self.classifier = classifier
        return classifier
