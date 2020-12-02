import numpy as np
import random

class Perceptron():

    def __init__(self, data_features, data_targets, learning_rate, bias):
        self.features = data_features
        self.targets = data_targets
        self.learning_rate = learning_rate
        self.bias = bias
        self.epochs = 0
        self.w = []

    
    def train(self, epochs):
        
        self.epochs = epochs

        w = np.array([random.random() for _ in range(len(self.features[0]))])

        for _ in range(self.epochs):
            for row_idx in range(len(self.features)):
                y = 0.
                for feature_idx in range(len(self.features[row_idx])):
                    y = y + self.features[row_idx][feature_idx] * w[feature_idx] 
                
                y = y + self.bias

                target = 1.0 if (y>0) else 0.

                delta = (self.targets[row_idx] - target)

                # Update weights
                if(delta):
                    for i in range(len(w)):
                        delta_w = self.learning_rate * self.features[feature_idx][i] * delta
                        w[i] = w[i] + delta_w

        self.w = w

    def classify(self, features):

        y = 0.
        for feature_idx in range(len(features)):
            y = y + features[feature_idx] * self.w[feature_idx] 

        output = 1.0 if (y>0) else 0.

        return output