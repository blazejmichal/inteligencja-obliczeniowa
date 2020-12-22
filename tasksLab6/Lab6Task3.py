import numpy
import pandas
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix


class Lab6Task3:
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    @classmethod
    def run(cls):
        print('Lab6Task3')
        cls.getSets()
        (y_true, y_predictions) = cls.runMLP()
        (scores, misses) = cls.drawConfusionMatrix(y_true, y_predictions, 'Neural Networks - MLP')
        cls.drawSummaryBarChart((scores, misses))
        cls.drawAccuracyBarChart(cls.countAccuracy((scores, misses)))
        pass

    @classmethod
    def runMLP(cls):
        mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)
        trainedMlp = mlp.fit(cls.X_train, cls.y_train)
        y_predictions = trainedMlp.predict(cls.X_test)
        y_predictions = cls.mapDataSet(y_predictions)
        y_true_values = cls.y_test.to_numpy()
        y_true_values = cls.mapDataSet(y_true_values)
        return (y_true_values, y_predictions)

    @classmethod
    def getSets(cls):
        dataFrame = pandas.read_csv('diabetes.csv')
        (x, y) = cls.getXandY(dataFrame, 8, 'class')
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
        cls.X_train = X_train
        cls.X_test = X_test
        cls.y_train = y_train
        cls.y_test = y_test
        pass

    @classmethod
    def drawConfusionMatrix(cls, y_true_values, y_prediction, algorithmName):
        confusionMatrix = confusion_matrix(y_true_values, y_prediction)
        print(confusionMatrix)
        labels = ['Chory', 'Zdrowy']
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(confusionMatrix)
        plt.title(algorithmName)
        fig.colorbar(cax)
        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
        positivePredictions = confusionMatrix[0][0] + confusionMatrix[1][1]
        negativePredictions = sum(confusionMatrix[0]) + sum(confusionMatrix[1]) - positivePredictions
        print(
            'Poprawnie sklasyfikowane: ' + str(positivePredictions) + '\n' + 'Zle sklasyfikowane: ' + str(
                negativePredictions))
        print('True Positive: ' + str(confusionMatrix[0][0]))
        print('True Negative: ' + str(confusionMatrix[1][1]))
        print('False Positive: ' + str(confusionMatrix[1][0]))
        print('False Negative: ' + str(confusionMatrix[0][1]))
        return (positivePredictions, negativePredictions)

    @classmethod
    def getXandY(cls, dataFrame, featureColumnAmount, resultColumnName):
        # bierze nazwy pierwszych kolumn okreslonych liczba
        features = list(dataFrame.columns[:featureColumnAmount])
        y = dataFrame[resultColumnName]
        x = dataFrame[features]
        return (x, y)

    @classmethod
    def mapDataSet(cls, dataSet):
        changes = {'tested_positive': 0, 'tested_negative': 1}
        return [changes.get(x, x) for x in dataSet]

    @classmethod
    def drawSummaryBarChart(cls, mlp):
        objects = ('MLP TP&TN', 'MLP FP&FN')
        y_pos = numpy.arange(len(objects))
        performance = [mlp[0], mlp[1]]
        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.title('Strzaly')
        plt.show()

    @classmethod
    def drawAccuracyBarChart(cls, mlp):
        x_objects = ['MLP']
        y_pos = numpy.arange(len(x_objects))
        performance = [mlp]
        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, x_objects)
        plt.title('Dokladnosc')
        plt.show()

    @classmethod
    def countAccuracy(cls, shots):
        return shots[0] / (sum(shots))
