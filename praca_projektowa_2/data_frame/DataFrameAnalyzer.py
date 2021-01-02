import matplotlib.pyplot as pyplot


class DataFrameAnalyzer:
    data_frame: object

    def analyze(self):
        for name, values in self.data_frame.iteritems():
            print('\n')
            print('Nazwa kolumny: ' + str(name))
            print('Wartosc minimalna: ' + str(min(values)))
            print('Wartosc maksymalna: ' + str(max(values)))
            print('Wartosc srednia: ' + str(sum(values) / len(values)))
            column = values.value_counts()
            column = column.sort_index(ascending=True)
            column = self.divide_to_ranges(column, 10)
            labels = [*column]
            values = column.values()
            pyplot.pie(values, labels=labels)
            pyplot.title("Zestawienie czestosci wystepowania wartosci w kolumnie: " + str(name))
            pyplot.show()
            print('\n')
            pass

    def divide_to_ranges(self, collection, window_size):
        result = {}
        if (len(collection) <= window_size):
            for index, value in collection.items():
                result[index] = value
            return result
        while len(collection) > window_size:
            range = collection.head(window_size)
            label = str(range.head(1).index[0]) + ' - ' + str(range.tail(1).index[0])
            value = sum(range)
            result[label] = value
            collection = collection.tail(len(collection) - window_size)
        label = str(collection.head(1).index[0]) + ' - ' + str(collection.tail(1).index[0])
        value = sum(collection)
        result[label] = value
        return result

    def __init__(self):
        pass

    @classmethod
    def build(cls, data_frame):
        data_frame_analyzer = DataFrameAnalyzer()
        data_frame_analyzer.data_frame = data_frame
        return data_frame_analyzer
