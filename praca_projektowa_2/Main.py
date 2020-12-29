from operator import attrgetter

from praca_projektowa_2.analyzers.KNeighboursAnalyzer import KNeighboursAnalyzer
from praca_projektowa_2.analyzers.NaiveBayesAnalyzer import NaiveBayesAnalyzer
from praca_projektowa_2.analyzers.NeuralNetworksMLPAnalyzer import NeuralNetworksMLPAnalyzer
from praca_projektowa_2.analyzers.TreeAnalyzer import TreeAnalyzer
from praca_projektowa_2.classifiers.KNeighboursClassifier import KNeighboursClassifier
from praca_projektowa_2.classifiers.NaiveBayesClassifier import NaiveBayesClassifier
from praca_projektowa_2.classifiers.NeuralNetworksMLPClassifier import NeuralNetworksMLPClassifier
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
    # naive_bayess_classifier = runNaiveBayessClassifier(x_train, x_test, y_train, y_test)
    naive_bayes_analyzer = runNaiveBayessAnalyzer(x_train, x_test, y_train, y_test)
    # tree_classifier = runTreeClassifier(x_train, x_test, y_train, y_test)
    tree_analyzer = runTreeAnalyzer(x_train, x_test, y_train, y_test)
    k_neighbours_analyzer = runKNeighboursAnalyzer(x_train, x_test, y_train, y_test)
    # k_neighbours_classifier = runKNeighboursClassifier(x_train, x_test, y_train, y_test)
    # neural_networks_mlp_classifier = runNeuralNetworksMLPClassifier(x_train, x_test, y_train, y_test)
    neural_networks_mlp_analyzer = runNeuralNetworksMLPAnalyzer(x_train, x_test, y_train, y_test)


# def runNaiveBayessClassifier(x_train, x_test, y_train, y_test):
#     naive_bayess_classifier = NaiveBayesClassifier.build(x_train, x_test, y_train, y_test)
#     naive_bayess_classifier.run()
#     return naive_bayess_classifier

def runNaiveBayessAnalyzer(x_train, x_test, y_train, y_test):
    naive_bayess_analyzer = NaiveBayesAnalyzer.build(x_train, x_test, y_train, y_test)
    naive_bayess_analyzer.run()
    return naive_bayess_analyzer


def runTreeAnalyzer(x_train, x_test, y_train, y_test):
    tree_analyzer = TreeAnalyzer.build(x_train, x_test, y_train, y_test)
    tree_analyzer.run()
    return tree_analyzer


def runTreeClassifier(x_train, x_test, y_train, y_test):
    tree_classifier = TreeClassifier.build(x_train, x_test, y_train, y_test)
    tree_classifier.run()
    return tree_classifier


def runKNeighboursAnalyzer(x_train, x_test, y_train, y_test):
    k_neighbours_analyzer = KNeighboursAnalyzer.build(x_train, x_test, y_train, y_test)
    k_neighbours_analyzer.run()
    return k_neighbours_analyzer

def runNeuralNetworksMLPAnalyzer(x_train, x_test, y_train, y_test):
    analyzer = NeuralNetworksMLPAnalyzer.build(x_train, x_test, y_train, y_test)
    analyzer.run()
    return analyzer
    # classifiers = []
    # for i in range(1, 5):
    #     for j in range(1, 5):
    #         hidden_layer_sizes = (i * 10, i * 10)
    #         max_iter = j * 1000
    #         classifier = NeuralNetworksMLPClassifier.build(x_train, x_test, y_train, y_test, hidden_layer_sizes,
    #                                                        max_iter)
    #         classifier.run()
    #         classifiers.append(classifier)
    # k_neighbours_classifier = max(classifiers, key=attrgetter('accuracy'))
    # return k_neighbours_classifier

# def runNeuralNetworksMLPClassifier(x_train, x_test, y_train, y_test):
#     classifiers = []
#     for i in range(1, 5):
#         for j in range(1, 5):
#             hidden_layer_sizes = (i * 10, i * 10)
#             max_iter = j * 1000
#             classifier = NeuralNetworksMLPClassifier.build(x_train, x_test, y_train, y_test, hidden_layer_sizes,
#                                                            max_iter)
#             classifier.run()
#             classifiers.append(classifier)
#     k_neighbours_classifier = max(classifiers, key=attrgetter('accuracy'))
#     return k_neighbours_classifier


if __name__ == '__main__':
    main()
