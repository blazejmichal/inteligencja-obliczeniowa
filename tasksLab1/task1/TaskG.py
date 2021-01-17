import numpy as np


class TaskG:

    def __init__(self):
        pass

    @classmethod
    def execute(cls, v):
        cls.normalizeVector(v)

    @classmethod
    def normalizeVector(cls, v):
        max = np.amax(v)
        min = np.amin(v)
        result = list(v)
        for i in range(len(v)):
            result[i] = (v[i] - min) / float(max - min)
        print('Znormalizowany wektor: ' + str(result))
        print('Max oryginalnego wektora: ' + str(max))
        maxIndex = v.index(max)
        print('Pozycja max: ' + str(maxIndex))
        print('Nowy element na tej pozycji: ' + str(float(result[maxIndex])))
