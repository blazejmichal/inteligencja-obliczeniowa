import pandas as pd
import difflib as difflib


class Task1C:
    flowerNames = ['Setosa', 'Versicolor', 'Virginica']

    @staticmethod
    def run():
        df = pd.read_csv('db/iris_with_errors.csv')

        for i in range(df['variety'].size):
            if (not (df['variety'][i] in Task1C.flowerNames)):
                df['variety'][i] = Task1C.repair(df['variety'][i])
        for i, j in df.iteritems():
            print(i)
            print(j)

    @staticmethod
    def repair(word):
        flowerName = difflib.get_close_matches(word, Task1C.flowerNames)
        return flowerName[0]
