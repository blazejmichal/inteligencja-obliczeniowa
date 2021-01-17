import matplotlib.pyplot as plt
import csv


class Task2c:

    def __init__(self):
        pass

    @staticmethod
    def execute():
        x = []
        y = []

        with open('miasta.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for column in reader:
                x.append(column['Rok'])
                y.append(column['Gdansk'])

        plt.plot(x, y, 'r', label='Krzywa wykresu')
        plt.xlabel('Lata')
        plt.ylabel('Liczba ludnosci [w tys.]')
        plt.title('Ludnosc w miastach Polski (Gdansk)')
        plt.legend()
        plt.show()
