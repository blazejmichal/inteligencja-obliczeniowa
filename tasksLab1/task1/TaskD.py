from random import randrange
import numpy as np


class TaskD:
    a = [[], [], []]
    b = [[], [], []]

    def __init__(self):
        for i in range(3):
            for j in range(3):
                self.a[i].append(randrange(9))
                self.b[i].append(randrange(9))

    def execute(self):
        self.multiplyCoordinates()
        self.multiplyMatrices()

    def multiplyCoordinates(self):
        result = [[], [], []]
        for i in range(3):
            for j in range(3):
                num = self.a[i][j] * self.b[i][j]
                result[i].append(num)
        print('Mnozenie macierzy po wspolrzednych: ' + str(result))
        print('Numpy: ' + str(np.multiply(self.a, self.b)))

    def multiplyMatrices(self):
        result = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    result[i][j] += self.a[i][k] * self.b[k][j]
        print('Mnozenie macierzy: ' + str(result))
        print('Numpy: ' + str(np.matmul(self.a, self.b)))
