import csv
import pandas as pd


class Task1A:

    @staticmethod
    def clean():
        print("cleaning")
        df = pd.read_csv('db/iris_with_errors.csv')
        print("znalezione suma: ")
        print(df.isnull().any(axis=1).sum())
        print("znalezione linie: ")
        print(df[df.isna().any(axis=1)])
        df = df.fillna("N/A")
        print("finished cleaning")
        print("znalezione suma: ")
        print(df.isnull().any(axis=1).sum())
        print("znalezione linie: ")
        print(df[df.isna().any(axis=1)])

    @staticmethod
    def execute():
        with open('db/iris_with_errors.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                print(row)
