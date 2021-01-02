import pandas as pandas
from pandas.core.dtypes.common import is_string_dtype, is_numeric_dtype
from sklearn.model_selection import train_test_split


class DataFrameLoader:
    result_column_name = 'quality'
    data_frame_location = 'data_frame/winequality-red.csv'
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    mapping_legend = {}
    data_frame = None

    def run(self):
        dataFrame = self.loadDataFrame()
        dataFrame = self.correctDataFrame(dataFrame)
        self.data_frame = dataFrame
        (x, y) = self.getXandY(dataFrame, self.result_column_name)
        self.divideDataFrame(x, y)
        return dataFrame

    def loadDataFrame(self):
        dataFrame = pandas.read_csv(self.data_frame_location)
        if ("id" in dataFrame):
            dataFrame.drop("id", axis=1, inplace=True)
        return dataFrame

    def divideDataFrame(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        pass

    def getXandY(self, dataFrame, resultColumnName):
        features = list(dataFrame.drop(resultColumnName, axis=1, inplace=False))
        y = dataFrame[resultColumnName]
        x = dataFrame[features]
        return (x, y)

    def correctDataFrame(self, data_frame):
        for col in data_frame.columns:
            if (is_string_dtype(data_frame[col])):
                data_frame[col] = self.mapColumn(data_frame[col])
                data_frame[col] = pandas.to_numeric(data_frame[col], errors='coerce')
        errors = data_frame[data_frame.isna().any(axis=1)]
        errors_amount = errors.shape[0]
        print("Ilość krotek zawierających nieprawidłowe dane: ")
        print(errors_amount)
        if (errors_amount > 0):
            data_frame = data_frame.dropna()
            # data_frame = data_frame.fillna(None)
        return data_frame

    def __init__(self):
        self.run()
        pass

    @classmethod
    def build(cls, result_column_name, data_frame_location):
        data_frame_loader = DataFrameLoader()
        data_frame_loader.result_column_name = result_column_name
        data_frame_loader.data_frame_location = data_frame_location
        data_frame_loader.run()
        return data_frame_loader

    def mapColumn(self, column):
        mapping_pattern = {}
        unique_values = column.unique()
        for i, unique_value in enumerate(unique_values):
            mapping_pattern[unique_value] = i
        self.mapping_legend[column.name] = mapping_pattern
        column = [mapping_pattern.get(x, x) for x in column]
        return column

    def get_result_mapping_legend(self):
        try:
            result_mapping_legend = self.mapping_legend[self.result_column_name]
        except KeyError:
            result_mapping_legend = {}
        return result_mapping_legend
