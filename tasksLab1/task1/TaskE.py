from random import randrange
import numpy as np


class TaskE:
    a = []

    def __init__(self):
        i = 0
        while i < 50:
            self.a.append(randrange(101))
            i += 1

    def execute(self):
        print('Wektor ' + str(len(self.a)) + ' losowych: ' + str(self.a))
        print('Numpy: ' + str(np.random.randint(1, 101, 50)))
        return self.a
