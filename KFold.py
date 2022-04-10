# Universidade Federal do Rio Grande do Sul
# Instituto de Informatica
# INF01017 - Aprendizado de Maquina (2021/2)
# Juliane da Rocha Alves and Gabriel Lando

import pandas as pd
import numpy as np
import random
from typing import List, Tuple
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from ConfusionMatrix import ConfusionMatrix


class CrossValidation:
    """
    This class implements the logic behind the Cross Validation. It will split in K folds the X (values) and y (target)
    keeping the proportion between the classes in the folds.

    This class was implemented for Classification problems. It receives a Scikit classifier and apply
    the cross validation method.
    """
    def __init__(self, classifier: BaseEstimator = None, k_folds: int = 5, X: np.ndarray = None, y: np.ndarray = None) -> None:
        self.classifier = classifier
        self.k_folds = k_folds
        self.X = X
        self.y = y
        self.accuracy = []
        self.precision = []
        self.recall = []
        self.specificity = []
        self.f1_measure = []

    def fit(self) -> None:
        num_samples = len(self.y)
        folds_size = round(num_samples / self.k_folds)
        # Get the frequency of each class
        target, frequency = np.unique(self.y, return_counts=True)

        class_proportion = {}
        class_index = {}
        for ind in range(len(target)):
            # Calculates the classes proportion
            class_proportion[target[ind]] = (frequency[ind]/num_samples)*100
            # Gets all indexes for each class in the dataset
            class_index[target[ind]] = np.where(self.y == target[ind])[0]

        indexes_used = []
        X_folds = []
        y_folds = []
        for k in range(self.k_folds):
            X_fold = []
            y_fold = []
            print(f"Generating fold {k+1}")
            if k != self.k_folds-1:
                for key in class_proportion.keys():
                    # Calculates how many elements it needs to have in each fold for each class, to keep the proportion
                    num_elements = round((class_proportion[key]*folds_size)/100)
                    # Gets random N (num_elements) elements from each class
                    ind = random.sample(class_index[key].tolist(), num_elements)

                    for i in ind:
                        # Verify if the index was already used
                        if i not in indexes_used:
                            X_fold.append(self.X[i].tolist())
                            y_fold.append(self.y[i][0])
                            indexes_used.append(i)
                        else:
                            # Calculates a new index if the index was already used
                            new_i = i
                            while new_i in indexes_used:
                                new_i = random.sample(class_index[key].tolist(), 1)[0]

                            X_fold.append(self.X[new_i].tolist())
                            y_fold.append(self.y[new_i][0])
                            indexes_used.append(new_i)
                # Append the fold
                X_folds.append(X_fold)
                y_folds.append(y_fold)
            else:
                # The last fold will have the rest of the values that are not in the previous folds
                all_indexes = [ind for ind in range(num_samples)]
                indexes_for_test = list(set(all_indexes) - set(indexes_used))
                for ind in indexes_for_test:
                    X_fold.append(self.X[ind].tolist())
                    y_fold.append(self.y[ind][0])
                # Append the last fold
                X_folds.append(X_fold)
                y_folds.append(y_fold)

        # Train the model K times (K = number of folds)
        for k in range(self.k_folds):
            X_train = []
            y_train = []
            X_test = []
            y_test = None

            # The K determines the fold to be used to test
            for values in X_folds[k]:
                X_test.append(values)
            y_test = y_folds[k]

            print(f"Using the fold {k} for test")

            # The j determines the folds to be used to train. The fold used to test is not used
            for j in range(self.k_folds):
                if j != k:
                    print(f"Using the fold {j} for train")
                    for values in X_folds[j]:
                        X_train.append(values)
                    for y in y_folds[j]:
                        y_train.append(y)

            # Gets the shape of the training set
            shape = self._get_train_fold_shape(k, X_folds)

            # Reshape the train and test sets (n_samples, n_features)
            X_train = np.array(X_train).reshape((shape, self.X.shape[1]))
            y_train = np.array(y_train).reshape((shape,))

            X_test = np.array(X_test).reshape((self.X.shape[0] - shape, self.X.shape[1]))
            y_test = np.array(y_test).reshape((self.X.shape[0] - shape,))

            self.classifier.fit(X_train, y_train)
            y_pred = self.classifier.predict(X_test)

            self.print_report(y_pred, y_test)

            print("*******************************")

        self.print_final_report()

    def print_report(self, y_pred: np.ndarray, y_test: np.ndarray) -> None:
        cm = ConfusionMatrix(y_test, y_pred)
        accuracy = cm.get_accuracy()
        precision = cm.get_precision()
        recall = cm.get_recall()
        specificity = cm.get_specificity()
        f1_measure = cm.get_f1_measure()
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"Specificity: {specificity}")
        print(f"F1_measure: {f1_measure}")
        self.accuracy.append(accuracy)
        self.precision.append(precision)
        self.recall.append(recall)
        self.specificity.append(specificity)
        self.f1_measure.append(f1_measure)
        cm.plot_confusion_matrix()

    def print_final_report(self) -> None:
        accuracy = self._get_avg(self.accuracy)
        precision = self._get_avg(self.precision)
        recall = self._get_avg(self.recall)
        specificity = self._get_avg(self.specificity)
        f1_measure = self._get_avg(self.f1_measure)

        print("****** FINAL REPORT ******")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"Specificity: {specificity}")
        print(f"F1_measure: {f1_measure}")

    def _get_train_fold_shape(self, k: int, X_folds: List) -> int:
        rows = 0
        for fold in range(self.k_folds):
            if fold != k:
                rows = rows + len(X_folds[fold])
        return rows

    def _get_avg(self, array: List) -> float:
        return sum(array) / len(array)


if __name__ == "__main__":
    spotify_df = pd.read_csv("dataset-of-10s.csv")
    features = ["danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness",
                         "instrumentalness", "liveness", "valence", "tempo", "duration_ms", "time_signature",
                         "chorus_hit", "sections"]
    possible_features = ["danceability", "energy", "instrumentalness", "duration_ms", "sections", "loudness"]
    X = spotify_df.loc[:, features].values
    y = spotify_df.loc[:, ["target"]].values

    print("*************************************")
    print("****** Running a Random Forest ******")
    print("*************************************")
    dtree = DecisionTreeClassifier(max_depth=10,
                                   max_features="sqrt",
                                   min_samples_leaf=10,
                                   min_samples_split=25,
                                   splitter="best",
                                   criterion="gini")

    cv = CrossValidation(classifier=dtree, k_folds=15, X=X, y=y)
    cv.fit()

    # It is necessary to normalize the values for K-NN and Logistic Regression
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    #
    print("***************************")
    print("****** Running a KNN ******")
    print("***************************")
    knn = KNeighborsClassifier(n_neighbors=13)
    cv = CrossValidation(classifier=knn, k_folds=15, X=X, y=y)
    cv.fit()

    print("*******************************************")
    print("****** Running a Logistic Regression ******")
    print("*******************************************")
    lr = LogisticRegression(tol=0.001, solver="newton-cg")
    cv = CrossValidation(classifier=lr, k_folds=15, X=X, y=y)
    cv.fit()
