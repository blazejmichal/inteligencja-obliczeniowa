from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


class NaiveBayesClassifier:
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    name = 'Naive Bayes'
    accuracy = 0
    y_test_values = []
    y_predictions = []

    def run(self):
        self.getValuesOfTestSet()
        classifier = self.train()
        self.predict(classifier)
        self.evaluateClassifier(self.y_test_values, self.y_predictions)
        pass

    def train(self):
        classifier = GaussianNB()
        classifier = classifier.fit(self.x_train, self.y_train)
        return classifier

    def predict(self, classifier):
        self.y_predictions = classifier.predict(self.x_test)
        pass

    def getValuesOfTestSet(self):
        self.y_test_values = self.y_test.to_numpy()
        pass

    def evaluateClassifier(self, y_test_values, y_predictions):
        self.accuracy = accuracy_score(y_predictions, y_test_values)
        pass

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

# class NaiveBayesClassifier:
#     x_train = []
#     x_test = []
#     y_train = []
#     y_test = []
#
#     def run(self):
#         print('Naive Bayess')
#         y_true_values = self.getTrueValuesOfTrainSet()
#         classificator = self.train()
#         y_predictions = self.predict(classificator)
#         (scores, misses) = self.drawConfusionMatrix(y_true_values, y_predictions)
#         self.drawSummaryBarChart((scores, misses))
#         self.drawAccuracyBarChart(self.countAccuracy((scores, misses)))
#         pass
#
#     def train(self):
#         naive_bayess_classificator = GaussianNB().fit(self.x_train, self.y_train)
#         return naive_bayess_classificator
#
#     def predict(self, classificator):
#         y_predictions = classificator.predict(self.x_test)
#         return y_predictions
#
#     def getTrueValuesOfTrainSet(self):
#         return self.y_test.to_numpy()
#
#     def drawConfusionMatrix(self, y_true_values, y_prediction):
#         confusionMatrix = confusion_matrix(y_true_values, y_prediction)
#         print(confusionMatrix)
#         labels = ['1', '2', '3', '4', '5']
#         fig = pyplot.figure()
#         ax = fig.add_subplot(111)
#         cax = ax.matshow(confusionMatrix)
#         pyplot.title('Naive Bayess')
#         fig.colorbar(cax)
#         ax.set_xticklabels([''] + labels)
#         ax.set_yticklabels([''] + labels)
#         pyplot.xlabel('Predicted')
#         pyplot.ylabel('True')
#         pyplot.show()
#         accuracy = accuracy_score(y_prediction, y_true_values)
#         print('Dokladnosc Naive Bayess')
#         print(accuracy)
#         positivePredictions = 0
#         for i in range(len(labels)):
#             positivePredictions += confusionMatrix[i][i]
#         negativePredictions = len(y_true_values) - positivePredictions
#         print(
#             'Poprawnie sklasyfikowane: ' + str(positivePredictions) + '\n' + 'Zle sklasyfikowane: ' + str(
#                 negativePredictions))
#         return (positivePredictions, negativePredictions)
#
#     def drawSummaryBarChart(self, algorithm):
#         objects = ('Naive Bayess trafione', 'Naive Bayess nietrafione')
#         y_pos = numpy.arange(len(objects))
#         performance = [algorithm[0], algorithm[1]]
#         pyplot.bar(y_pos, performance, align='center', alpha=0.5)
#         pyplot.xticks(y_pos, objects)
#         pyplot.title('Strzaly')
#         pyplot.show()
#
#     def drawAccuracyBarChart(self, algorithm):
#         x_objects = ['Naive Bayess']
#         y_pos = numpy.arange(len(x_objects))
#         performance = [algorithm]
#         pyplot.bar(y_pos, performance, align='center', alpha=0.5)
#         pyplot.xticks(y_pos, x_objects)
#         pyplot.title('Dokladnosc')
#         pyplot.show()
#
#     def countAccuracy(self, shots):
#         return shots[0] / (sum(shots))
#
#     def __init__(self):
#         pass
#
#     @classmethod
#     def build(cls, x_train, x_test, y_train, y_test):
#         algorithm = NaiveBayesClassifier()
#         algorithm.x_train = x_train
#         algorithm.x_test = x_test
#         algorithm.y_train = y_train
#         algorithm.y_test = y_test
#         return algorithm
