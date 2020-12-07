import pandas as pd

from tasksLab3.IrisColumn import IrisColumn


class Task2B:

    @classmethod
    def run(cls):
        df = pd.read_csv('db/iris.csv')
        varianceSepalLengths = cls.calculateVariance(df, 'sepal.length')
        varianceSepalWidths = cls.calculateVariance(df, 'sepal.width')
        variancePetalLengths = cls.calculateVariance(df, 'petal.length')
        variancePetalWidths = cls.calculateVariance(df, 'petal.width')

        totalVariances = sum([varianceSepalLengths, varianceSepalWidths, variancePetalLengths, variancePetalWidths])
        sepalLengthsColumn = IrisColumn(varianceSepalLengths, 'sepal.length')
        sepalWidthsColumn = IrisColumn(varianceSepalWidths, 'sepal.width')
        petalLengthsColumn = IrisColumn(variancePetalLengths, 'petal.length')
        petalWidthsColumn = IrisColumn(variancePetalWidths, 'petal.width')
        irisColumns = [sepalLengthsColumn, sepalWidthsColumn, petalLengthsColumn, petalWidthsColumn]
        irisColumns.sort()

        usedVariances = 0.0
        usedColumns = []
        for irisColumn in irisColumns:
            usedColumns.append(irisColumn)
            usedVariances += irisColumn.variance
            profit = usedVariances / totalVariances
            if (profit >= 0.8):
                cls.printResult(usedColumns)
                return usedColumns
        pass

    @classmethod
    def printResult(cls, usedColumns):
        print('Kolumny, które powinny być użyte:')
        [print(col.name) for col in usedColumns]

    @classmethod
    def calculateVariance(cls, df, columnName):
        columnValues = df.loc[:, [columnName]].values
        variance = pd.np.var(columnValues)
        return variance
