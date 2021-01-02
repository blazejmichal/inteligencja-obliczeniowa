from operator import attrgetter
from praca_projektowa_2.analyzers.Analyzer import Analyzer
from praca_projektowa_2.classifiers.KNeighboursClassifier import KNeighboursClassifier


class KNeighboursAnalyzer(Analyzer):
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
        analyzer = KNeighboursAnalyzer()
        analyzer.x_train = x_train
        analyzer.x_test = x_test
        analyzer.y_train = y_train
        analyzer.y_test = y_test
        analyzer.result_mapping_legend = result_mapping_legend
        return analyzer

    def findBestClassifier(self):
        classifiers = []
        for i in range(1, 15):
            classifier = KNeighboursClassifier.build(self.x_train, self.x_test, self.y_train, self.y_test, i)
            classifier.run()
            classifiers.append(classifier)
        classifier = max(classifiers, key=attrgetter('accuracy'))
        self.classifier = classifier
        return classifier
