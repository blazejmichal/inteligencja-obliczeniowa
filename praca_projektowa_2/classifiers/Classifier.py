from sklearn.metrics import accuracy_score


class Classifier:
    x_train: list
    x_test: list
    y_train: list
    y_test: list
    name: str
    accuracy: int
    y_test_values: list
    y_predictions: list

    def train(self):
        pass

    def run(self):
        self.getValuesOfTestSet()
        classifier = self.train()
        self.predict(classifier)
        self.evaluateClassifier(self.y_test_values, self.y_predictions)
        pass

    def predict(self, classifier):
        self.y_predictions = classifier.predict(self.x_test)
        pass

    def getValuesOfTestSet(self):
        self.y_test_values = self.y_test.to_numpy()
        pass

    def evaluateClassifier(self, y_test_values, y_predictions):
        self.accuracy = accuracy_score(y_predictions, y_test_values)
        pass
