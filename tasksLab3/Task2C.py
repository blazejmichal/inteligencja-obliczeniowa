import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class Task2C:

    @staticmethod
    def run():
        df = pd.read_csv('db/iris.csv')
        features = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']
        x = df.loc[:, features].values
        y = df.loc[:, ['variety']].values
        x = StandardScaler().fit_transform(x)
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data=principalComponents
                                   , columns=['principal component 1', 'principal component 2'])
        finalDf = pd.concat([principalDf, df[['variety']]], axis=1)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        ax.set_title('2 component PCA', fontsize=20)
        targets = ['Setosa', 'Versicolor', 'Virginica']
        colors = ['r', 'g', 'b']
        for target, color in zip(targets, colors):
            indicesToKeep = finalDf['variety'] == target
            ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                       , finalDf.loc[indicesToKeep, 'principal component 2']
                       , c=color
                       , s=50)
        ax.legend(targets)
        ax.grid()
        plt.show()
