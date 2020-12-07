import timeit
import matplotlib.pyplot as plt


class BackpackGreedyAlgorithm:
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
    def getBestSet(cls, items, capacity):
        xIteration = []
        yValue = []
        cls.startTime = timeit.default_timer()
        bestSet = []
        items.sort(reverse=True)
        totalValue = 0
        iteration = 0
        for item in items:
            currentWeight = int(item.weight)
            currentValue = int(item.value)
            iteration += 1
            xIteration.append(iteration)
            if capacity - currentWeight >= 0:
                capacity -= currentWeight
                totalValue += currentValue
                yValue.append(totalValue)
                bestSet.append(item)
            else:
                fraction = capacity / currentWeight
                totalValue += currentValue * fraction
                capacity = int(capacity - (currentWeight * fraction))
                yValue.append(totalValue)
                break
        cls.endTime = timeit.default_timer()
        cls.plotChart(xIteration, yValue)
        return bestSet

    """
    Rysuje wykres z proby
    """

    @classmethod
    def plotChart(cls, x, y):
        plt.title("Greedy")
        plt.plot(x, y, label='Przebieg wartosci')
        plt.xlabel('Iteracje')
        plt.ylabel('Wartosci')
        plt.legend()
        plt.show()

    """
    Wyswietla info z proby w console log'u.
    """

    @classmethod
    def printInfo(cls, time, bestSet):
        print("\n")
        print("Algorytm Greedy")
        print("Czas: " + str(time))
        print("Znaleziony zestaw: " + str(bestSet))
        print("\n")

    """
    Ta metoda takze zawiera logike tego algorytmu. 
    Jest uruchamiana w celu znalezienia maksymalnej mozliwej wartosci wypelnionego plecaka. 
    Dzieki temu algorytm genetyczny "wie w ktorej generacji znalazl maksymalna wartosc"
    """

    @classmethod
    def getMaxValue(cls, items, capacity):
        items.sort(reverse=True)
        totalValue = 0
        for item in items:
            currentWeight = int(item.weight)
            currentValue = int(item.value)
            if capacity - currentWeight >= 0:
                capacity -= currentWeight
                totalValue += currentValue
            else:
                fraction = capacity / currentWeight
                totalValue += currentValue * fraction
                capacity = int(capacity - (currentWeight * fraction))
                break
        return totalValue
