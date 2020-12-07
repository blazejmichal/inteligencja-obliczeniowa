"""
Klasa modelowa przedstawiajaca obiekt wkladany do plecaka
"""


class Item(object):

    def __init__(self, value=0, weight=0, name=None):
        self.value = value
        self.weight = weight
        self.name = name

    def __lt__(self, other):
        selfCost = self.value // self.weight
        otherCost = other.value // other.weight
        return selfCost < otherCost

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()
