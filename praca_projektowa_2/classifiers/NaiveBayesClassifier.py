from sklearn.naive_bayes import GaussianNB
from praca_projektowa_2.classifiers.Classifier import Classifier


class NaiveBayesClassifier(Classifier):
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    name = 'Naive Bayes'
    accuracy = 0
    y_test_values = []
    y_predictions = []

    def __init__(self):
        pass

    @classmethod
    def build(cls, x_train, x_test, y_train, y_test):
        classifier = NaiveBayesClassifier()
        classifier.x_train = x_train
        classifier.x_test = x_test
        classifier.y_train = y_train
        classifier.y_test = y_test
        classifier.run()
        return classifier

    def train(self):
        classifier = GaussianNB()
        classifier = classifier.fit(self.x_train, self.y_train)
        return classifier
