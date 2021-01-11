import pandas
from apyori import apriori
from matplotlib import pyplot


def main():
    print("lab7")
    runTask()
    pass


def runTask():
    print("Lab7Task2")
    dataFrame = pandas.read_csv("titanic.csv", header=None)
    dataFrame = dataFrame.drop(dataFrame.columns[0], axis=1)
    dataFrame.shape
    transactions = []
    for i in range(0, dataFrame.shape[0]):
        transactions.append([str(dataFrame.values[i, j]) for j in range(0, len(dataFrame.columns))])
    # dziala niestety tylko na python 2.x
    rules = apriori(transactions, min_confidence=0.8, min_support=0.005)
    results = list(rules)
    resultsPoint2 = pandas.DataFrame(results)
    # Podpunkt 2
    print('\nZnalezione wszystkie zasady: ')
    print('Ilosc wszystkich zasad: ' + str(len(resultsPoint2)))
    print(resultsPoint2)
    df = pandas.DataFrame(columns=('Items', 'Antecedent', 'Consequent', 'Support', 'Confidence', 'Lift'))
    Support = []
    Confidence = []
    Lift = []
    Items = []
    Antecedent = []
    Consequent = []
    for RelationRecord in results:
        for ordered_stat in RelationRecord.ordered_statistics:
            Support.append(RelationRecord.support)
            Items.append(RelationRecord.items)
            Antecedent.append(ordered_stat.items_base)
            Consequent.append(ordered_stat.items_add)
            Confidence.append(ordered_stat.confidence)
            Lift.append(ordered_stat.lift)
    df['Items'] = list(map(set, Items))
    df['Antecedent'] = list(map(set, Antecedent))
    df['Consequent'] = list(map(set, Consequent))
    df['Support'] = Support
    df['Confidence'] = Confidence
    df['Lift'] = Lift
    df.sort_values(by='Confidence', ascending=False, inplace=True)
    print('\nPosortowane wzgledem nawjiekszej ufnosci')
    print(df)
    # Podpunkt 3
    print('Zasady gdzie przezyli: ')
    survived = df.set_index('Items').filter(regex='Yes', axis=0)
    print('Dlugosc: ' + str(len(survived)))
    print(survived)
    print('Zasady gdzie nie przezyli: ')
    not_survived = df.set_index('Items').filter(regex='No', axis=0)
    print('Dlugosc: ' + str(len(not_survived)))
    print(not_survived)
    labels=["Wszystkie reguly", "Reguly z przezyli", "Reguly z nie przezyli"]
    values = [len(resultsPoint2),len(survived), len(not_survived)]
    pyplot.barh(labels, values)
    for index, value in enumerate(values):
        pyplot.text(value, index, str(value))
    pyplot.title('Podsumowanie: ')
    pyplot.show()
    pass


if __name__ == '__main__':
    main()
