# Zadanie 3
# W załączonym zbiorze danych diabetes.csv znajdują się dane kobiet indiańskiego pochodzenia
# z USA, które zachorowały lub nie zachorowały na cukrzycę. Klasyfikator ma na celu
# diagnozowanie choroby na podstawie parametrów medycznych kobiety. Sprawdź jak działają
# poznane klasyfikatory na tej bazie danych. Dokonaj porównania:
# • k-NN, k=3
# • k-NN, k=5
# • k-NN, k=11
# • Naiwny bayesowski.
# • Drzewa decyzyjne.
# W rozwiązaniu zadania uwzględnij następujące punkty:
# a) Podziel w losowy sposób bazę danych na zbiór treningowy (67%) i testowy (33%).
# b) Uruchom każdy z klasyfikatorów wykorzystując paczki i dokonaj ewaluacji ma zbiorze
# testowym wyświetlając procentową dokładność i macierz błędu. Przydatne linki:
# Naive Bayes:
# https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn
# https://scikit-learn.org/stable/modules/naive_bayes.html
# k-NN:
# https://towardsdatascience.com/k-nearest-neighbor-python-2fccc47d2a55
# https://scikit-learn.org/stable/modules/neighbors.html
# c) Nanieś wszystkie dokładności klasyfikatorów na wykres słupkowy. Każdy słupek
# odpowiada jednemu klasyfikatorowi, a wysokość słupka to jego dokładność
# procentowa. Jeśli trzeba to dodaj legendę.
# d) Pytanie dodatkowe:
# Chcemy zminimalizować błędy, gdy klasyfikator chore osoby klasyfikuje jako zdrowe ( i
# odsyła do domu bez leków). Który z klasyfikatorów najbardziej się do tego nadaje?
from sklearn import tree

import pandas
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as numpy
from sklearn.neighbors import KNeighborsClassifier


class Lab5Task3:
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    @classmethod
    def run(cls):
        cls.getSets()

        (bayes_y_true, bayes_y_predictions) = cls.runNaiveBayes()
        (bayes_scores, bayes_misses) = cls.drawConfusionMatrix(bayes_y_true, bayes_y_predictions, 'Bayes')

        (tree_y_true, tree_y_predictions) = cls.runTree()
        (tree_scores, tree_misses) = cls.drawConfusionMatrix(tree_y_true, tree_y_predictions, 'Tree')

        (three_neighbours_true, three_neighbours_y_predictions) = cls.runNeighbours(3)
        (three_neighbours_scores, three_neighbours_misses) = cls.drawConfusionMatrix(three_neighbours_true,
                                                                                     three_neighbours_y_predictions,
                                                                                     '3KNN')

        (five_neighbours_true, five_neighbours_y_predictions) = cls.runNeighbours(5)
        (five_neighbours_scores, five_neighbours_misses) = cls.drawConfusionMatrix(five_neighbours_true,
                                                                                   five_neighbours_y_predictions,
                                                                                   '5KNN')

        (eleven_neighbours_true, eleven_neighbours_y_predictions) = cls.runNeighbours(11)
        (eleven_neighbours_scores, eleven_neighbours_misses) = cls.drawConfusionMatrix(eleven_neighbours_true,
                                                                                       eleven_neighbours_y_predictions,
                                                                                       '11KNN')

        cls.drawSummaryBarChart((bayes_scores, bayes_misses),
                                (tree_scores, tree_misses),
                                (three_neighbours_scores, three_neighbours_misses),
                                (five_neighbours_scores, five_neighbours_misses),
                                (eleven_neighbours_scores, eleven_neighbours_misses))
        cls.drawAccuracyBarChart(cls.countAccuracy((bayes_scores, bayes_misses)),
                                 cls.countAccuracy((tree_scores, tree_misses)),
                                 cls.countAccuracy((three_neighbours_scores, three_neighbours_misses)),
                                 cls.countAccuracy((five_neighbours_scores, five_neighbours_misses)),
                                 cls.countAccuracy((eleven_neighbours_scores, eleven_neighbours_misses)))
        pass

    @classmethod
    def runNaiveBayes(cls):
        gnb = GaussianNB().fit(cls.X_train, cls.y_train)
        y_predictions = gnb.predict(cls.X_test)
        y_predictions = cls.mapDataSet(y_predictions)
        y_true_values = cls.y_test.to_numpy()
        y_true_values = cls.mapDataSet(y_true_values)
        print('\nAlgorytm naiwny Bayesowski')
        return (y_true_values, y_predictions)

    @classmethod
    def runTree(cls):
        treeClassifier = tree.DecisionTreeClassifier()
        treeClassifier = treeClassifier.fit(cls.X_train, cls.y_train)
        y_predictions = treeClassifier.predict(cls.X_test)
        y_predictions = cls.mapDataSet(y_predictions)
        y_true_values = cls.y_test.to_numpy()
        y_true_values = cls.mapDataSet(y_true_values)
        print('Drzewo decyzyjne')
        return (y_true_values, y_predictions)

    @classmethod
    def runNeighbours(cls, times):
        kNeighborsClassifier = KNeighborsClassifier(n_neighbors=times, metric='euclidean')
        kNeighborsClassifier.fit(cls.X_train, cls.y_train)
        y_predictions = kNeighborsClassifier.predict(cls.X_test)
        y_predictions = cls.mapDataSet(y_predictions)
        y_true_values = cls.y_test.to_numpy()
        y_true_values = cls.mapDataSet(y_true_values)
        print(str(times) + ' najblizszych sasiadow')
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
    def drawSummaryBarChart(cls, bayes, tree, threeKNN, fiveKNN, elevenKNN):
        objects = (
            'Bayes TP&TN', 'Bayes FP&FN', 'Tree TP&TN', 'Tree FP&FN', '3KNN TP&TN', '3KNN FP&FN', '5KNN TP&TN',
            '5KNN FP&FN', '11KNN TP&TN', '11KNN FP&FN')
        y_pos = numpy.arange(len(objects))
        performance = [bayes[0], bayes[1], tree[0], tree[1], threeKNN[0], threeKNN[1], fiveKNN[0], fiveKNN[1],
                       elevenKNN[0], elevenKNN[1]]
        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.title('Strzaly')
        plt.show()

    @classmethod
    def drawAccuracyBarChart(cls, bayes, tree, threeKNN, fiveKNN, elevenKNN):
        x_objects = ('Bayes', 'Tree', '3KNN', '5KNN', '11KNN')
        y_pos = numpy.arange(len(x_objects))
        performance = [bayes, tree, threeKNN, fiveKNN, elevenKNN]
        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, x_objects)
        plt.title('Dokladnosc')
        plt.show()

    @classmethod
    def countAccuracy(cls, shots):
        return shots[0] / (sum(shots))
