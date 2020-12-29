from sklearn import tree

from praca_projektowa_2.classifiers.Classifier import Classifier


class TreeClassifier(Classifier):
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    name = 'Tree'
    accuracy = 0
    y_test_values = []
    y_predictions = []

    def __init__(self):
        pass

    @classmethod
    def build(cls, x_train, x_test, y_train, y_test):
        classifier = TreeClassifier()
        classifier.x_train = x_train
        classifier.x_test = x_test
        classifier.y_train = y_train
        classifier.y_test = y_test
        classifier.run()
        return classifier

    def train(self):
        classifier = tree.DecisionTreeClassifier()
        classifier = classifier.fit(self.x_train, self.y_train)
        return classifier
