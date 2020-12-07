import matplotlib.pyplot as plt
import itertools
import timeit


class BackpackBruteForce:
    ITEMS = []
    CAPACITY = 0
    startTime = 0
    endTime = 0

    """
    Glowna metoda uruchamiajaca algorytm
    """

    @classmethod
    def run(cls, ITEMS, CAPACITY):
        cls.setInput(ITEMS, CAPACITY)
        bestSet = cls.getBestSet(ITEMS, CAPACITY)
        time = cls.endTime - cls.startTime
        cls.printInfo(time, bestSet)
        return time

    """
    Przypisuje do pol wartosci argumentow przekazanych z metody uruchamiajacej projekt
    """

    @classmethod
    def setInput(cls, ITEMS, CAPACITY):
        cls.ITEMS = ITEMS
        cls.CAPACITY = CAPACITY

    """
    Metoda z zawarta logika algorytmu. Zwraca zestaw przedmiotow o maksymalnej wartosci w danych przypadku plecaka
    """

    @classmethod
    def getBestSet(cls, ITEMS, CAPACITY):
        xIteration = []
        yValue = []
        cls.startTime = timeit.default_timer()
        iteration = 0
        matchingSets = []
        for i in range(0, len(ITEMS) + 1):
            for subSet in itertools.combinations(ITEMS, i):
                subSetValue = cls.getValueOfSet(subSet)
                subSetWeights = map(lambda item: item.weight, subSet)
                subSetWeight = sum(subSetWeights)
                iteration += 1
                if (subSetWeight <= CAPACITY):
                    xIteration.append(iteration)
                    yValue.append(subSetValue)
                    matchingSets.append(subSet)
        maxValue = max(matchingSets, key=lambda matchingSet: cls.getValueOfSet(matchingSet))
        cls.endTime = timeit.default_timer()
        cls.plotChart(xIteration, yValue)
        return maxValue

    """
    Rysuje wykres z proby
    """

    @classmethod
    def plotChart(cls, x, y):
        plt.title("Brute Force")
        plt.plot(x, y, label='Przebieg wartosci')
        plt.xlabel('Iteracje')
        plt.ylabel('Wartosci')
        plt.legend()
        plt.show()

    """
    Mapuje kolekcje Itemow na kolekcje ich wartosci. Po czym sumuje by uzyskac laczna wartosc zestawu przedmiotow.
    """

    @classmethod
    def getValueOfSet(cls, set):
        setValues = map(lambda item: item.value, set)
        return sum(setValues)

    """
    Wyswietla info z proby w console log'u.
    """

    @classmethod
    def printInfo(cls, time, bestSet):
        print("\n")
        print("Algorytm Brute Force")
        print("Czas: " + str(time))
        print("Znaleziony zestaw: " + str(bestSet))
        print("\n")
