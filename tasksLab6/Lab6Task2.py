from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


class Lab6Task2:

    @classmethod
    def run(cls):
        print("\nLab6Task2")
        iris = load_iris()
        datasets = train_test_split(iris.data, iris.target, test_size=0.2)
        train_data, test_data, train_labels, test_labels = datasets
        # normalizedIrisData = preprocessing.normalize(iris.data)
        # Skalowanie zbioru danych
        scaler = StandardScaler()
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
        # Klasyfikator Multi Layer Perceptron
        mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)
        # labels to kolumna z wynikami (jaki gatunek kwiatu)
        # uczenie klasyfikatora
        mlp.fit(train_data, train_labels)
        # wyniki klasyfikatora na treningowym
        predictions_train = mlp.predict(train_data)
        print('\nDokladnosc na zbiorze treningowym')
        print(accuracy_score(predictions_train, train_labels))
        # wyniki klasyfikatora na testowym
        predictions_test = mlp.predict(test_data)
        print('\nDokladnosc na zbiorze testowym')
        print(accuracy_score(predictions_test, test_labels))
        print('\nMacierz bledow na zbiorze treningowym')
        print(confusion_matrix(predictions_train, train_labels))
        print('\nMacierz bledow na zbiorze testowym')
        print(confusion_matrix(predictions_test, test_labels))
        print('\nRaport klasyfikacyjny')
        print(classification_report(predictions_test, test_labels))
