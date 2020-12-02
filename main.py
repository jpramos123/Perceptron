from sklearn import datasets
from Perceptron import Perceptron
import numpy as np
from sklearn import metrics

if __name__ == "__main__":
    
    # Dados separados em 70% treino e 30% teste
    
    iris = datasets.load_iris(return_X_y=True)

    iris_features = iris[0][:100]
    iris_targets = iris[1][:100]

    #Dados de treinamento
    iris_features_train = np.concatenate((iris_features[:35], iris_features[50:85]))
    iris_targets_train = np.concatenate((iris_targets[:35], iris_targets[50:85]))

    #Dados de teste
    iris_features_test = np.concatenate((iris_features[35:50], iris_features[85:]))
    iris_targets_test = np.concatenate((iris_targets[35:50], iris_targets[85:]))


    ptn = Perceptron(iris_features_train, iris_targets_train, learning_rate=1e-2, bias=0)

    ptn.train(epochs=1000)

    pred_arr = []
    for features in iris_features_test:
        pred_arr.append(ptn.classify(features))
    
    # Metrica de score da biblioteca skitlearn
    print("FINAL SCORE: ", metrics.accuracy_score(iris_targets_test, pred_arr))