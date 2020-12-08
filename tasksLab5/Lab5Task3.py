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
import pandas
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt


class Lab5Task3:

    @classmethod
    def run(cls):
        # (trainSet, testSet) = cls.divideDataFrame()
        (bayes_y_true, bayes_y_predictions,) = cls.runNaiveBayes()
        cls.drawConfusionMatrix(bayes_y_true, bayes_y_predictions)
        pass

    @classmethod
    def drawConfusionMatrix(cls, y_true_values, y_prediction):
        confusionMatrix = confusion_matrix(y_true_values, y_prediction)
        print(confusionMatrix)
        labels = ['Chory', 'Zdrowy']
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(confusionMatrix)
        plt.title('Confusion matrix of the classifier')
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
        pass

    @classmethod
    def runNaiveBayes(cls):
        dataFrame = pandas.read_csv('diabetes.csv')
        # X, y = load_iris(return_X_y=True)
        (x, y) = cls.getXandY(dataFrame, 8, 'class')
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
        gnb = GaussianNB().fit(X_train, y_train)
        y_predictions = gnb.predict(X_test)
        y_predictions = cls.mapListToBoolean(y_predictions)
        y_true_values = y_test.to_numpy()
        y_true_values = cls.mapListToBoolean(y_true_values)
        print('Algorytm naiwny Bayesowski')
        return (y_true_values, y_predictions)

    @classmethod
    def getXandY(cls, dataFrame, featureColumnAmount, resultColumnName):
        # bierze nazwy pierwszych kolumn okreslonych liczba
        features = list(dataFrame.columns[:featureColumnAmount])
        y = dataFrame[resultColumnName]
        x = dataFrame[features]
        return (x, y)

    @classmethod
    def mapListToBoolean(cls, list):
        changes = {'tested_positive': 0, 'tested_negative': 1}
        return [changes.get(x, x) for x in list]
