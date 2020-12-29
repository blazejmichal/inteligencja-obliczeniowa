import numpy
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix
from praca_projektowa_2.classifiers.TreeClassifier import TreeClassifier


class TreeAnalyzer:
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    classifier = None

    def findBestClassifier(self):
        classifier = TreeClassifier.build(self.x_train, self.x_test, self.y_train, self.y_test)
        self.classifier = classifier
        return classifier

    def run(self):
        best_classifier = self.findBestClassifier()
        y_test_values = best_classifier.y_test_values
        y_predictions = best_classifier.y_predictions
        print(best_classifier.name)
        (scores, misses) = self.drawConfusionMatrix(y_test_values, y_predictions)
        self.drawSummaryBarChart((scores, misses))
        self.drawAccuracyBarChart(self.countAccuracy((scores, misses)))
        pass

    def drawConfusionMatrix(self, y_true_values, y_prediction):
        confusionMatrix = confusion_matrix(y_true_values, y_prediction)
        print(confusionMatrix)
        labels = ['1', '2', '3', '4', '5']
        fig = pyplot.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(confusionMatrix)
        pyplot.title(self.classifier.name)
        fig.colorbar(cax)
        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)
        pyplot.xlabel('Predicted')
        pyplot.ylabel('True')
        pyplot.show()
        print('Dokladnosc ' + self.classifier.name)
        print(self.classifier.accuracy)
        positivePredictions = 0
        for i in range(len(labels)):
            positivePredictions += confusionMatrix[i][i]
        negativePredictions = len(y_true_values) - positivePredictions
        print(
            'Poprawnie sklasyfikowane: ' + str(positivePredictions) + '\n' + 'Zle sklasyfikowane: ' + str(
                negativePredictions))
        return (positivePredictions, negativePredictions)

    def drawSummaryBarChart(self, algorithm):
        objects = (self.classifier.name + ' trafione', self.classifier.name + ' nietrafione')
        y_pos = numpy.arange(len(objects))
        performance = [algorithm[0], algorithm[1]]
        pyplot.bar(y_pos, performance, align='center', alpha=0.5)
        pyplot.xticks(y_pos, objects)
        pyplot.title('Strzaly')
        pyplot.show()

    def drawAccuracyBarChart(self, algorithm):
        x_objects = [self.classifier.name]
        y_pos = numpy.arange(len(x_objects))
        performance = [algorithm]
        pyplot.bar(y_pos, performance, align='center', alpha=0.5)
        pyplot.xticks(y_pos, x_objects)
        pyplot.title('Dokladnosc')
        pyplot.show()

    def countAccuracy(self, shots):
        return shots[0] / (sum(shots))

    def __init__(self):
        pass

    @classmethod
    def build(cls, x_train, x_test, y_train, y_test):
        analyzer = TreeAnalyzer()
        analyzer.x_train = x_train
        analyzer.x_test = x_test
        analyzer.y_train = y_train
        analyzer.y_test = y_test
        return analyzer