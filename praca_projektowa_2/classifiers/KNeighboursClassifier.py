from sklearn.neighbors import KNeighborsClassifier

from praca_projektowa_2.classifiers.Classifier import Classifier


class KNeighboursClassifier(Classifier):
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    k = 1
    name = str(k) + ' najblizszych sasiadow'
    accuracy = 0
    y_test_values = []
    y_predictions = []

    def __init__(self):
        pass

    @classmethod
    def build(cls, x_train, x_test, y_train, y_test, k):
        classifier = KNeighboursClassifier()
        classifier.x_train = x_train
        classifier.x_test = x_test
        classifier.y_train = y_train
        classifier.y_test = y_test
        classifier.k = k
        classifier.name = str(k) + ' najblizszych sasiadow'
        return classifier

    def train(self):
        classifier = KNeighborsClassifier(n_neighbors=self.k, metric='euclidean')
        classifier = classifier.fit(self.x_train, self.y_train)
        return classifier
