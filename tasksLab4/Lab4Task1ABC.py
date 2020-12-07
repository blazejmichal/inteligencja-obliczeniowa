import numpy as numpy
import pandas as pandas


# a) Napisz w Pythonie funkcję, która na podstawie czterech numerycznych
# parametrów irysa odgadnie jego gatunek wykorzystując do tego jedynie dwie
# instrukcje if-else. Schemat pseudokodowy:
# myPredictRow(sl,sw,pl,pw) {
#  if (...) {
#  return(...)
#  } else {
#  if (...) {
#  return(...)
#  } else {
#  return(...)
#  }
#  }
# }
# W miejsce kropek należy wpisać warunki na sl,sw,pl,pw np. (sl>3.5 lub
# podobne), a return musi dawać jedną z wartości: versicolor, virginica, setosa.
# Jakie warunki wpisać? Po prostu spróbuj zgadnąć patrząc na bazę danych. Może
# masz wprawne oko i uda ci się dostrzec jakieś zależności w danych.
# Zauważ, że jak wyciągasz wnioski na podstawie wszystkich 150 rekordów, to
# zbiór treningowy to cała baza danych.

class Lab4Task1ABC:

    @classmethod
    def run(cls):
        result = cls.predict()
        print('Task1A: ' + str(result * 100) + '%')

    @classmethod
    def predictGenre(cls, sepalLength, sepalWidth, petalLength, petalWidth):
        if (sepalLength < 5.5):
            return "Setosa"
        else:
            if petalLength < 5:
                return "Versicolor"
            else:
                return "Virginica"

    @classmethod
    def predict(cls):
        dataFrame = pandas.read_csv('iris.csv')
        copied = dataFrame.copy()
        predictColumn = 'prediction'
        copied[predictColumn] = [cls.predictGenre(sepalLength, sepalWidth, petalLength, petalWidth) for
                                 sepalLength, sepalWidth, petalLength, petalWidth in
                                 zip(dataFrame['sepal.length'], dataFrame['sepal.width'], dataFrame['petal.length'],
                                     dataFrame['petal.width'])]
        return len(numpy.where(copied['variety'] == copied['prediction'])[0]) / len(copied)
