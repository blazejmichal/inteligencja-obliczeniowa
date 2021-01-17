from random import randrange
import numpy as np


class TaskF:

    def __init__(self):
        pass

    @classmethod
    def execute(cls, v):
        cls.calcAverage(v)

    @classmethod
    def calcAverage(cls, v):
        print('Srednia wektora: ' + str(np.average(v)))
        print('Min.: ' + str(np.min(v)))
        print('Max.: ' + str(np.max(v)))
        print('Odchylenie standardowe.: ' + str(np.std(v)))