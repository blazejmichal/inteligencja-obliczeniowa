from praca_projektowa_2.algorithms.NaiveBayesAlgorithm import NaiveBayesAlgorithm
from praca_projektowa_2.data_frame.DataFrameLoader import DataFrameLoader


def main():
    print("Praca projektowa 2")
    data_frame_loader = DataFrameLoader()
    # data_frame_loader = DataFrameLoader.build(1, 'test', 'test')
    # data_frame = data_frame_loader.run()
    x_train = data_frame_loader.x_train
    x_test = data_frame_loader.x_test
    y_train = data_frame_loader.y_train
    y_test = data_frame_loader.y_test
    naive_bayess_algorithm = NaiveBayesAlgorithm.build(x_train, x_test, y_train, y_test)
    naive_bayess_algorithm.run()


if __name__ == '__main__':
    main()
