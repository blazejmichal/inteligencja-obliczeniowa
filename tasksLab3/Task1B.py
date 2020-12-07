import pandas as pd
import numpy as np


class Task1B:

    @staticmethod
    def run():
        columnNames = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']
        df = pd.read_csv('db/iris_with_errors.csv')
        for columnName in columnNames:
            df[columnName] = pd.to_numeric(df[columnName], errors='coerce').fillna(0)
            median = df[columnName].median()
            df[columnName] = np.where(
                (df[columnName] <= 0) | (df[columnName] >= 15), median,
                df[columnName])
            for i, j in df.iteritems():
                print(i)
                print(j)
