from TaskB import TaskB
import math
import numpy as np


class TaskC:
    a = [3, 8, 9, 10, 12]
    b = [8, 7, 7, 5, 6]

    @staticmethod
    def execute():
        TaskC.calculateScalarProduct()
        TaskC.calculateDistance()

    @staticmethod
    def calculateScalarProduct():
        v = TaskB.multiplyVectors()
        result = 0
        for i in range(len(v)):
            result += v[i]
        print('Iloczyn skalarny: ' + str(result))
        print('Numpy: ' + str(np.dot(TaskC.a, TaskC.b)))

    @staticmethod
    def calculateDistance():
        result = math.sqrt(
            ((TaskC.b[0] - TaskC.a[0]) ** 2)
            +
            ((TaskC.b[1] - TaskC.a[1]) ** 2)
            +
            ((TaskC.b[2] - TaskC.a[2]) ** 2)
            +
            ((TaskC.b[3] - TaskC.a[3]) ** 2)
            +
            ((TaskC.b[4] - TaskC.a[4]) ** 2)
        )
        print('Odleglosc euklidesowa: ' + str(result))

        print('Numpy: ' + str(np.linalg.norm(np.subtract(TaskC.a, TaskC.b))))
