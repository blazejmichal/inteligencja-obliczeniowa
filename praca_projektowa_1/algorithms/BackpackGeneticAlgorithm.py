import random
import timeit
import numpy as numpy
import matplotlib.pyplot as plt


class BackpackGeneticAlgorithm:
    ITEMS = []
    MAX_VALUE = 0
    CAPACITY = 0
    POPULATION_SIZE = 50
    GENERATION_AMOUNT = 100
    MUTATION_CHANCE = 0.5

    """
    Glowna metoda uruchamiajaca algorytm. Zawiera takze jego logike.
    """

    @classmethod
    def run(cls, ITEMS, CAPACITY, MAX_VALUE):
        yMax = []
        yAverage = []
        xGenerations = []
        cls.setInput(ITEMS, CAPACITY)
        start = timeit.default_timer()
        generation = 0
        population = cls.createStartingPopulation(cls.POPULATION_SIZE)
        for i in range(0, cls.GENERATION_AMOUNT):
            fitnessValues = cls.getFitnessValues(population)
            cls.updateX(xGenerations, generation)
            cls.updateYMax(yMax, fitnessValues)
            cls.updateYAverage(yAverage, fitnessValues)
            population = cls.evolvePopulation(population)
            generation += 1
            if MAX_VALUE in fitnessValues:
                break
        stop = timeit.default_timer()
        time = stop - start
        cls.plotChart(yMax, yAverage, xGenerations)
        cls.printInfo(time)
        return time

    """
    Przypisuje do pol wartosci argumentow przekazanych z metody uruchamiajacej projekt
    """

    @classmethod
    def setInput(cls, ITEMS, CAPACITY):
        cls.ITEMS = ITEMS
        cls.CAPACITY = CAPACITY

    """
    Tworzy poczatkowa populacje. Opiera to na losowaniu liczb.
    """

    @classmethod
    def createStartingPopulation(cls, size):
        return [cls.createChromosome() for i in range(0, size)]

    """
    Liczy fitness dla kazdego chromosomu w populacji. Zwraca to w postaci tablicy.
    """

    @classmethod
    def getFitnessValues(cls, population):
        fitnessValues = []
        for chromosome in population:
            fitnessValues.append(cls.calculateFitness(chromosome))
        return fitnessValues

    """
    Metoda licza fitness dla chromosmu. 
    """

    @classmethod
    def calculateFitness(cls, chromosome):
        chromosomeValue = 0
        chromosomeWeight = 0
        for i in range(len(chromosome)):
            if (chromosome[i] == 1):
                chromosomeValue += cls.ITEMS[i].value
                chromosomeWeight += cls.ITEMS[i].weight
        return 0 if chromosomeWeight > cls.CAPACITY else chromosomeValue

    """
    Wystawia populacje na ewolucje. 
    """

    @classmethod
    def evolvePopulation(cls, population):
        parents = cls.chooseParents(population)
        parents = cls.mutateRandomParents(parents)
        children = cls.createChildren(parents)
        parents.extend(children)
        return parents

    """
    Tworzy chromosom losujac jego elementy.
    """

    @classmethod
    def createChromosome(cls):
        return [random.randint(0, 1) for i in range(0, len(cls.ITEMS))]

    """
    Podmienia wylosowany element chromosomu na wartosc przeciwna
    """

    @classmethod
    def mutateRandomBit(cls, chromosome):
        randomInt = random.randint(0, len(chromosome) - 1)
        if chromosome[randomInt] == 1:
            chromosome[randomInt] = 0
        else:
            chromosome[randomInt] = 1

    """
    Sortuje kolekcje chromosmow. Chromosmu z najwyzszym fitness ustawiane sa na poczatku kolekcji
    """

    @classmethod
    def putBestChromosomesAsFirst(cls, population):
        population = sorted(population, key=lambda chromosome: cls.calculateFitness(chromosome), reverse=True)
        return population

    """
    Tworzy dzieci na podstawie rodzicow. 
    """

    @classmethod
    def createChildren(cls, parents):
        children = []
        desired_length = cls.POPULATION_SIZE - len(parents)
        while len(children) < desired_length:
            child = cls.createChild(parents)
            children.append(child)
        return children

    """
    Losuje rodzicow. Dzieli ich na pol i laczy w postac dziecka.
    Dziecko jest poddawane mutacji.
    """

    @classmethod
    def createChild(cls, parents):
        parent0 = parents[random.randint(0, len(parents) - 1)]
        parent1 = parents[random.randint(0, len(parents) - 1)]
        half = int(len(parent0) / 2)
        child = parent0[:half] + parent1[half:]
        child = cls.mutateChromosome(child)
        return child

    """
    Wybiera najbardziej wartosciowych rodzicow z populacji
    """

    @classmethod
    def chooseParents(cls, population):
        population = cls.putBestChromosomesAsFirst(population)
        parentEligibility = 0.3
        parentLength = int(parentEligibility * len(population))
        parents = population[:parentLength]
        return parents

    """
    Mutuje losowych rodzicow.
    """

    @classmethod
    def mutateRandomParents(cls, parents):
        for parent in parents:
            parent = cls.mutateChromosome(parent)
        return parents

    """
    Podaje chromosom mutacji.
    """

    @classmethod
    def mutateChromosome(cls, chromosome):
        if cls.MUTATION_CHANCE > random.random():
            cls.mutateRandomBit(chromosome)
        return chromosome

    """
    Wyswietla wykres
    """

    @classmethod
    def plotChart(cls, yFitness, yAverage, x):
        plt.plot(x, yFitness, 'r', label='Przebieg wartosci max')
        plt.plot(x, yAverage, 'b', label='Przebieg sredniej')
        plt.xlabel('Generacje')
        plt.ylabel('Wartosci')
        plt.legend()
        plt.show()

    """
    Dodaje wartosci do osi
    """

    @classmethod
    def updateX(cls, xGenerations, generation):
        xGenerations.append(generation)

    """
    Dodaje wartosci do osi
    """

    @classmethod
    def updateYMax(cls, yMax, fitnessValues):
        maxFitness = numpy.max(fitnessValues)
        yMax.append(maxFitness)

    """
    Dodaje wartosci do osi
    """

    @classmethod
    def updateYAverage(cls, yAverage, fitnessValues):
        averageFitness = numpy.average(fitnessValues)
        yAverage.append(averageFitness)

    """
    Wyswietla info z proby w console log'u.
    """

    @classmethod
    def printInfo(cls, time):
        print("\n")
        print("Algorytm Genetic")
        print("Czas: " + str(time))
        print("\n")
