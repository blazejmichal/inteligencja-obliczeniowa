from operator import attrgetter

from praca_projektowa_2.classifiers.KNeighboursClassifier import KNeighboursClassifier
from praca_projektowa_2.classifiers.NaiveBayesClassifier import NaiveBayesClassifier
from praca_projektowa_2.classifiers.TreeClassifier import TreeClassifier
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
    naive_bayess_classifier = runNaiveBayessClassifier(x_train, x_test, y_train, y_test)
    tree_classifier = runTreeClassifier(x_train, x_test, y_train, y_test)
    k_neighbours_classifier = runKNeighboursClassifier(x_train, x_test, y_train, y_test)


def runNaiveBayessClassifier(x_train, x_test, y_train, y_test):
    naive_bayess_classifier = NaiveBayesClassifier.build(x_train, x_test, y_train, y_test)
    naive_bayess_classifier.run()
    return naive_bayess_classifier


def runTreeClassifier(x_train, x_test, y_train, y_test):
    tree_classifier = TreeClassifier.build(x_train, x_test, y_train, y_test)
    tree_classifier.run()
    return tree_classifier


def runKNeighboursClassifier(x_train, x_test, y_train, y_test):
    k_neighbours_classifiers = []
    for i in range(1, 15):
        k_neighbours_classifier = KNeighboursClassifier.build(x_train, x_test, y_train, y_test, i)
        k_neighbours_classifier.run()
        k_neighbours_classifiers.append(k_neighbours_classifier)
    k_neighbours_classifier = max(k_neighbours_classifiers, key=attrgetter('accuracy'))
    return k_neighbours_classifier


if __name__ == '__main__':
    main()
