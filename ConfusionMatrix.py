# Universidade Federal do Rio Grande do Sul
# Instituto de Informatica
# INF01017 - Aprendizado de Maquina (2021/2)
# Juliane da Rocha Alves and Gabriel Lando

import numpy as np

class ConfusionMatrix:

    def __init__(self, true, pred):
        self._true = true
        self._pred = pred
        self._calculate_confusion_matrix()

    def _calculate_confusion_matrix(self):
        self._num = len(np.unique(self._true)) 
        result = np.zeros((self._num, self._num))


        for i in range(len(self._true)):
            result[self._true[i]][self._pred[i]] += 1

        self._tp = result[1,1]
        self._tn = result[0,0]
        self._fp = result[0,1]
        self._fn = result[1,0]
        self._matrix = result

    def get_confusion_matrix(self):
        return self._matrix

    def plot_confusion_matrix(self):
        print("            | Classe predita      ")
        print("----------------------------------")
        print("            |     |   +   |   -   ")
        print("Classe      |  +  | %5d | %5d " % (self._tp, self._fn))
        print("verdadeira  |  -  | %5d | %5d \n" % (self._fp, self._tn))

    def get_accuracy(self):
        return (self._tp + self._tn) / self._num

    def get_precision(self):
        return self._tp / (self._tp + self._fp)

    def get_recall(self):
        return self._tp / (self._tp + self._fn)

    def get_specificity(self):
        return self._tn / (self._tn + self._fp)

    def get_f1_measure(self):
        return self._tp / (self._tp + 0.5 * (self._fp + self._fn)) 
            
if __name__ == "__main__":
    true = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    pred = np.array([0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    cm = ConfusionMatrix(true, pred)

    cm.plot_confusion_matrix()

    print("Acuracia: %f" % (cm.get_accuracy()))
    print("Precisao: %f" % (cm.get_precision()))
    print("Recall: %f" % (cm.get_recall()))
    print("F1-measure: %f" % (cm.get_f1_measure()))
