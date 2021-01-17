import numpy as np


class TaskB:
    a = [3, 8, 9, 10, 12]
    b = [8, 7, 7, 5, 6]

    @staticmethod
    def execute():
        TaskB.addVectors()
        TaskB.multiplyVectors()

    @staticmethod
    def addVectors():
        result = []
        for i in range(len(TaskB.a)):
            result.append(TaskB.a[i] + TaskB.b[i])
        print('Suma wektorow: ' + str(result))
        print('Numpy: ' + str(np.add(TaskB.a, TaskB.b)))

    @staticmethod
    def multiplyVectors():
        result = []
        for i in range(len(TaskB.a)):
            result.append(TaskB.a[i] * TaskB.b[i])
        print('Iloczyn wektorow: ' + str(result))
        print('Numpy: ' + str(np.multiply(TaskB.a, TaskB.b)))
        return result
