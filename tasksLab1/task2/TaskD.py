import matplotlib.pyplot as plt
import csv


class Task2d:

    def __init__(self):
        pass

    @staticmethod
    def execute():
        x = []
        gdansk = []
        poznan = []
        szczecin = []

        with open('miasta.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for column in reader:
                x.append(column['Rok'])
                gdansk.append(column['Gdansk'])
                poznan.append(column['Poznan'])
                szczecin.append(column['Szczecin'])

        plt.plot(x, gdansk, 'r', label='Gdansk')
        plt.plot(x, poznan, 'b', label='Poznan')
        plt.plot(x, szczecin, 'g', label='Szczecin')
        plt.xlabel('Lata')
        plt.ylabel('Liczba ludnosci [w tys.]')
        plt.title('Ludnosc w miastach Polski')
        plt.legend()
        plt.show()
