from zadanie.ItemTask import ItemTask
from zadanie.BackpackGeneticAlgorithmTask import BackpackGeneticAlgorithmTask


def main():
    CAPACITY = 25
    ITEMS = initilizeItems()
    BackpackGeneticAlgorithmTask.run(ITEMS, CAPACITY)


def initilizeItems():
    ITEMS = [ItemTask() for i in range(11)]
    ITEMS[0] = ItemTask(100, 7, "zegar")
    ITEMS[1] = ItemTask(300, 7, "obraz-pejzaz")
    ITEMS[2] = ItemTask(200, 6, "obraz-portret")
    ITEMS[3] = ItemTask(40, 2, "radio")
    ITEMS[4] = ItemTask(500, 5, "laptop")
    ITEMS[5] = ItemTask(70, 6, "lampka nocna")
    ITEMS[6] = ItemTask(100, 1, "srebrne sztucce")
    ITEMS[7] = ItemTask(250, 3, "porcelana")
    ITEMS[8] = ItemTask(300, 10, "figura z brazu")
    ITEMS[9] = ItemTask(280, 3, "skorzana torebka")
    ITEMS[10] = ItemTask(300, 15, "odkurzacz")
    return ITEMS


if __name__ == "__main__":
    main()
