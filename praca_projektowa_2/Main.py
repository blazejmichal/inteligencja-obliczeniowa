from matplotlib import pyplot
from praca_projektowa_2.analyzers.KNeighboursAnalyzer import KNeighboursAnalyzer
from praca_projektowa_2.analyzers.NaiveBayesAnalyzer import NaiveBayesAnalyzer
from praca_projektowa_2.analyzers.NeuralNetworksMLPAnalyzer import NeuralNetworksMLPAnalyzer
from praca_projektowa_2.analyzers.TreeAnalyzer import TreeAnalyzer
from praca_projektowa_2.classifiers.TreeClassifier import TreeClassifier
from praca_projektowa_2.data_frame.DataFrameAnalyzer import DataFrameAnalyzer
from praca_projektowa_2.data_frame.DataFrameLoader import DataFrameLoader
from textwrap import wrap


def main():
    print("Praca projektowa 2")
    run_on_dataframe('quality', 'data_frame/winequality-red.csv')
    run_on_dataframe('Response', 'data_frame/healthinsurance.csv')


def run_on_dataframe(result_column_name, data_frame_location):
    data_frame_loader = DataFrameLoader.build(result_column_name, data_frame_location)
    runDataFrameAnalyzer(data_frame_loader)
    x_train = data_frame_loader.x_train
    x_test = data_frame_loader.x_test
    y_train = data_frame_loader.y_train
    y_test = data_frame_loader.y_test
    result_mapping_legend = data_frame_loader.get_result_mapping_legend()
    naive_bayes_analyzer = runNaiveBayessAnalyzer(x_train, x_test, y_train, y_test, result_mapping_legend)
    tree_analyzer = runTreeAnalyzer(x_train, x_test, y_train, y_test, result_mapping_legend)
    k_neighbours_analyzer = runKNeighboursAnalyzer(x_train, x_test, y_train, y_test, result_mapping_legend)
    neural_networks_mlp_analyzer = runNeuralNetworksMLPAnalyzer(x_train, x_test, y_train, y_test, result_mapping_legend)
    analyzers = [naive_bayes_analyzer, tree_analyzer, k_neighbours_analyzer, neural_networks_mlp_analyzer]
    runSummary(analyzers)


def runDataFrameAnalyzer(data_frame_loader):
    data_frame_analyzer = DataFrameAnalyzer.build(data_frame_loader.data_frame, data_frame_loader.mapping_legend)
    data_frame_analyzer.analyze()
    pass


def runNaiveBayessAnalyzer(x_train, x_test, y_train, y_test, result_mapping_legend):
    naive_bayess_analyzer = NaiveBayesAnalyzer.build(x_train, x_test, y_train, y_test, result_mapping_legend)
    naive_bayess_analyzer.run()
    return naive_bayess_analyzer


def runTreeAnalyzer(x_train, x_test, y_train, y_test, result_mapping_legend):
    tree_analyzer = TreeAnalyzer.build(x_train, x_test, y_train, y_test, result_mapping_legend)
    tree_analyzer.run()
    return tree_analyzer


def runTreeClassifier(x_train, x_test, y_train, y_test, result_mapping_legend):
    tree_classifier = TreeClassifier.build(x_train, x_test, y_train, y_test, result_mapping_legend)
    tree_classifier.run()
    return tree_classifier


def runKNeighboursAnalyzer(x_train, x_test, y_train, y_test, result_mapping_legend):
    k_neighbours_analyzer = KNeighboursAnalyzer.build(x_train, x_test, y_train, y_test, result_mapping_legend)
    k_neighbours_analyzer.run()
    return k_neighbours_analyzer


def runNeuralNetworksMLPAnalyzer(x_train, x_test, y_train, y_test, result_mapping_legend):
    analyzer = NeuralNetworksMLPAnalyzer.build(x_train, x_test, y_train, y_test, result_mapping_legend)
    analyzer.run()
    return analyzer


def runSummary(analyzers):
    labels = list(map(lambda analyzer: analyzer.classifier.name, analyzers))
    labels = ['\n'.join(wrap(label, 20)) for label in labels]
    values = list(map(lambda analyzer: analyzer.classifier.accuracy, analyzers))
    pyplot.barh(labels, values)
    pyplot.ylabel('Klasyfikatory')
    pyplot.xlabel('Dokladnosc')
    for index, value in enumerate(values):
        pyplot.text(value, index, str("{:.2%}".format(value)))
    pyplot.title('Podsumowanie: ')
    pyplot.show()
    pass


if __name__ == '__main__':
    main()
