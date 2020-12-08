# Zadanie 2
# W poprzednim zadaniu stworzyliśmy małe binarne drzewo decyzyjne postaci:
# Musieliśmy jednak stworzyć to drzewo sami. Są jednak algorytmy, takie jak ID3 czy
# C4.5, które tworzą takie drzewa automatycznie i to z o wiele większą precyzją niż
# człowiek. W Pythonie można skorzystać z paczki sklearn (tree):
# https://scikit-learn.org/stable/modules/tree.html,
# lub https://medium.com/@haydar_ai/learning-data-science-day-21-decision-tree-oniris-dataset-267f3219a7fa
# Wykorzystując wiedzę z samouczków wykonaj następujące polecenia.
# a) Podziel w losowy sposób bazę danych irysów na zbiór treningowy i zbiór
# testowy w proporcjach 70%/30%. Wyświetl oba zbiory.
# b) Wytrenuj drzewo decyzyjne na zbiorze treningowym.
# c) Wyświetl drzewo w formie tekstowej i w formie graficznej.
# d) Dokonaj ewaluacji klasyfikatora: sprawdź jak drzewo poradzi sobie z rekordami
# ze zbioru testowego. Wyświetl procent poprawnych odpowiedzi.
# e) Wyświetl macierz błędu (confusion matrix) dla tej ewaluacji. Wyjaśnij jakie błędy
# popełniał klasyfikator wskazując na liczby w macierzy błędu.

import pandas
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.tree import tree, export_text
import matplotlib.pyplot as plt
from sklearn import tree


class Lab4Task2:

    @classmethod
    def run(cls):
        dataFrame = pandas.read_csv('iris.csv')
        print("* iris types:", dataFrame['variety'].unique(), sep="\n")
        sets = cls.runA()
        trainSet = sets[0]
        testSet = sets[1]
        clf = cls.runB(trainSet)
        cls.runC(clf)
        (trueValues, predictions) = cls.runD(clf, testSet)
        cls.runE(trueValues, predictions)
        pass

    @classmethod
    def runA(cls):
        dataFrame = pandas.read_csv('iris.csv')
        trainSet = dataFrame.sample(frac=0.7, random_state=0)
        testSet = dataFrame.drop(trainSet.index)
        print('Dlugosc treningowego: ' + str(trainSet.size))
        print('Dlugosc testujacego: ' + str(testSet.size))
        print('Jest 100%: ' + str(trainSet.size + testSet.size == dataFrame.size))
        return (trainSet, testSet)

    @classmethod
    def runB(cls, trainSet):
        df_mod = cls.mapVarietyColumn(trainSet)
        # bierze nazwy 4 pierwszych kolumn
        features = list(df_mod.columns[:4])
        y = df_mod["variety"]
        x = df_mod[features]
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(x, y)
        return clf

    @classmethod
    def runC(cls, clf):
        # Graf
        tree.plot_tree(clf)
        plt.show()
        # Tekst
        dataFrame = pandas.read_csv('iris.csv')
        features = list(dataFrame.columns[:4])
        r = export_text(clf, feature_names=features)
        print(r)
        pass

    @classmethod
    def runD(cls, classifier, testSet):
        testSet = cls.mapVarietyColumn(testSet)
        testSetArguments = testSet.iloc[:, : 4]
        # Zostawia tylko ostatnio kolumne
        trueValues = testSet.iloc[:, -1:]
        # Trzeba odwrocic kolejnosc, bo klasyfikator sortuje w jakis sposob rezultaty
        trueValues = trueValues[::-1]
        predictions = classifier.predict(testSetArguments)
        accuracy = accuracy_score(trueValues, predictions)
        percentage = "{0:.0%}".format(accuracy)
        print("Skuteczność drzewa: " + str(percentage))
        return (trueValues, predictions)

    @classmethod
    def runE(cls, trueValues, predictions):
        confusionMatrix = confusion_matrix(trueValues, predictions)
        print(confusionMatrix)
        positivePredictions = confusionMatrix[0][0] + confusionMatrix[1][1] + confusionMatrix[2][2]
        negativePredictions = sum(confusionMatrix[0]) + sum(confusionMatrix[1]) + sum(
            confusionMatrix[2]) - positivePredictions
        print(
            'Poprawnie sklasyfikowane: ' + str(positivePredictions) + '\n' + 'Zle sklasyfikowane: ' + str(
                negativePredictions))
        pass

    @classmethod
    def mapVarietyColumn(cls, dataFrame):
        df_mod = dataFrame.copy()
        targets = df_mod['variety'].unique()
        map_to_int = {name: n for n, name in enumerate(targets)}
        # Dodanie kolumny z zmapowanymi na int wartosciami z kolumny variety
        df_mod["variety"] = df_mod['variety'].replace(map_to_int)
        return df_mod
