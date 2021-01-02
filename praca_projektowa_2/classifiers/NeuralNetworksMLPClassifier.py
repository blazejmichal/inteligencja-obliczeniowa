from sklearn.neural_network import MLPClassifier

from praca_projektowa_2.classifiers.Classifier import Classifier


class NeuralNetworksMLPClassifier(Classifier):
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    hidden_layer_sizes = (20, 20)
    max_iter = 2000
    name = 'Neural Networks - MLP'
    accuracy = 0
    y_test_values = []
    y_predictions = []

    def __init__(self):
        pass

    @classmethod
    def build(cls, x_train, x_test, y_train, y_test, hidden_layer_sizes, max_iter):
        classifier = NeuralNetworksMLPClassifier()
        classifier.x_train = x_train
        classifier.x_test = x_test
        classifier.y_train = y_train
        classifier.y_test = y_test
        classifier.hidden_layer_sizes = hidden_layer_sizes
        classifier.max_iter = max_iter
        classifier.name = 'Neural Networks - MLP' + ' (rozmiary ukrytych warstw = ' + str(
            hidden_layer_sizes) + ', maksymalna iteracja = ' + str(max_iter) + ')'
        return classifier

    def train(self):
        classifier = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, max_iter=self.max_iter)
        classifier = classifier.fit(self.x_train, self.y_train)
        return classifier
