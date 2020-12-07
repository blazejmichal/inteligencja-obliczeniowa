from projekt.model.Item import Item
from random import randrange


class ItemsInitilizer:
    """
    Sluzy do generowania inputu o wielkosci w zaleznosci od argumentu
    """

    @staticmethod
    def initilizeItemsGivenTimes(size):
        ITEMS = []
        for i in range(0, size):
            itemName = str("Item" + str(i))
            ITEMS.append(Item(randrange(10, 310, 10), randrange(1, 16, 1), itemName))
        return ITEMS
