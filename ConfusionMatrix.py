# Universidade Federal do Rio Grande do Sul
# Instituto de Informatica
# INF01017 - Aprendizado de Maquina (2021/2)
# Juliane da Rocha Alves and Gabriel Lando

import numpy as np

class ConfusionMatrix:
    """
    This class implements the metrics calculation. 

    During the class initialization, it will calculate the confusion matrix and will store it internally.
    After that, all metrics will be available calling their respective methods.
    """

    def __init__(self, true, pred):
        self._true = true
        self._pred = pred
        self._calculate_confusion_matrix()

    # Calculate the confusion matrix using a N x N array as input,
    # based on the array used to initialize the class.
    # The final result will be a M x M array, based on the quantity
    # of different values. (E.g.: in a binary class, it will be a 2x2 array)
    def _calculate_confusion_matrix(self):
        k = len(np.unique(self._true)) 
        result = np.zeros((k, k))

        for i in range(len(self._true)):
            result[self._true[i]][self._pred[i]] += 1

        # Store each result in its own variable
        # to be accessible easily inside this class
        self._tp = result[1,1] # True Positive
        self._tn = result[0,0] # True Negative
        self._fp = result[0,1] # False Positive
        self._fn = result[1,0] # False Negative
        self._matrix = result # The confusion matrix
        self._num = len(self._true) # Quantity of elements

    # Return the confusion matrix, if necessary
    def get_confusion_matrix(self):
        return self._matrix

    # Plot the confusion matrix using a predefined template
    def plot_confusion_matrix(self):
        print("      |     Predicted class ")
        print("------|---------------------")
        print("      |     |   +   |   -   ")
        print("True  |  +  | %5d | %5d " % (self._tp, self._fn))
        print("class |  -  | %5d | %5d \n" % (self._fp, self._tn))

    # Calculate and return the accuracy
    def get_accuracy(self):
        return (self._tp + self._tn) / self._num

    # Calculate and return the precision
    def get_precision(self):
        return self._tp / (self._tp + self._fp)

    # Calculate and return the recall
    def get_recall(self):
        return self._tp / (self._tp + self._fn)

    # Calculate and return the specicicity
    def get_specificity(self):
        return self._tn / (self._tn + self._fp)

    # Calculate and return the F1-measure
    def get_f1_measure(self):
        return self._tp / (self._tp + 0.5 * (self._fp + self._fn)) 
            
if __name__ == "__main__":

    # Using an example array to test
    true = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    pred = np.array([0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    # Instantiate the ConfusionMatrix class with the true and predicted arrays
    cm = ConfusionMatrix(true, pred)

    cm.plot_confusion_matrix()

    print("Acuracia: %f" % (cm.get_accuracy()))
    print("Precisao: %f" % (cm.get_precision()))
    print("Recall: %f" % (cm.get_recall()))
    print("F1-measure: %f" % (cm.get_f1_measure()))
