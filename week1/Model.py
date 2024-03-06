import pandas as pd
import numpy as np

class Model:
    numberOfClasses             = 0
    numberOfFeatures            = 0
    numberOfHiddenLayerUnits    = 0
    
    X = None
    Y = None
    
    X_forTraining   = None
    X_forTesting    = None
    Y_forTraining   = None
    Y_forTesting    = None
    Y_train_encoded = None
    Y_test_encoded  = None
    
    Weights1    = None
    bias1       = None
    Weights2    = None
    bias2       = None
    
    splitRange      = 75
    learningRate    = 0.01
    epochs          = 1000

    # One-hot encode the labels for categorical cross-entropy
    def one_hot_encode(self, labels, num_classes):
        return np.eye(num_classes)[labels]

    def prepareInputData(
        self,
        inputDataFrame: pd.DataFrame,
        numberOfClasses: int,
        numberOfFeatures: int,
        numberOfHiddenLayerUnits: int,
        splitRange: int,
        learningRate,
        epochs: int
    ):
        self.numberOfClasses            = numberOfClasses
        self.numberOfFeatures           = numberOfFeatures
        self.numberOfHiddenLayerUnits   = numberOfHiddenLayerUnits
        self.splitRange                 = splitRange

        self.X = inputDataFrame.iloc[:, 0:numberOfFeatures]
        self.Y = inputDataFrame.iloc[:, -1]

        self.X_forTraining, self.X_forTesting = (
            self.X.iloc[: self.splitRange],
            self.X.iloc[self.splitRange :],
        )
        self.Y_forTraining, self.Y_forTesting = (
            self.Y.iloc[: self.splitRange],
            self.Y.iloc[self.splitRange :],
        )

        self.Y_train_encoded = self.one_hot_encode(
            self.Y_forTraining, self.numberOfClasses
        )
        self.Y_test_encoded = self.one_hot_encode(
            self.Y_forTesting, self.numberOfClasses
        )

    def initializeStartRandomParameters(self):
        np.random.seed(42)
        self.Weights1 = np.random.randn(self.numberOfFeatures, self.numberOfHiddenLayerUnits)
        self.bias1 = np.zeros((1, self.numberOfHiddenLayerUnits))
        self.Weights2 = np.random.randn(self.numberOfHiddenLayerUnits, self.numberOfClasses)
        self.bias2 = np.zeros((1, self.numberOfClasses))

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return Z > 0

    def softmax(self, Z):
        Z = Z.to_numpy()
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def categoricalCrossEntropyLoss(self, Y, Y_hat, m):
        return -np.sum(Y * np.log(Y_hat + 1e-9)) / m
    
    def compute_loss(self, Y, Y_hat):
        m = Y.shape[0]
        loss = self.categoricalCrossEntropyLoss(self, Y, Y_hat, m)
        return loss

    def forwardPropagation(self, Learning_set_X):
        LinearTranformation_Z1 = Learning_set_X.dot(self.Weights1) + self.bias1
        ActivationFunction_A1 = self.relu(LinearTranformation_Z1)
        
        LinearTranformation_Z2 = ActivationFunction_A1.dot(self.Weights2) + self.bias2
        ActivationFunction_A2 = self.softmax(LinearTranformation_Z2)
        
        return {
            "LinearTranformation_Z1":   LinearTranformation_Z1, 
            "ActivationFunction_A1":    ActivationFunction_A1, 
            "LinearTranformation_Z2":   LinearTranformation_Z2, 
            "ActivationFunction_A2":    ActivationFunction_A2
        }

    def backwardPropagation(self, Learning_set_X, GroundTruth_Y_encoded, forwardPropagationResults):
        m = Learning_set_X.shape[0]
        
        dLinearTranformation_Z2 = forwardPropagationResults["ActivationFunction_A2"] - GroundTruth_Y_encoded
        
        dWeights2 = (1 / m) * np.dot(forwardPropagationResults["ActivationFunction_A1"].T, dLinearTranformation_Z2)
        
        dbias2 = (1 / m) * np.sum(dLinearTranformation_Z2, axis=0, keepdims=True)
        
        dLinearTranformation_Z1 = np.dot(dLinearTranformation_Z2, self.Weights2.T) * self.relu_derivative(forwardPropagationResults["LinearTranformation_Z1"])
        
        dWeights1 = (1 / m) * np.dot(Learning_set_X.T, dLinearTranformation_Z1)
        
        dbias1 = (1 / m) * np.sum(dLinearTranformation_Z1.to_numpy(), axis=0, keepdims=True)

        return {
            "dWeights1":    dWeights1, 
            "dbias1":       dbias1, 
            "dWeights2":    dWeights2, 
            "dbias2":       dbias2
        }

    def updateWeightsAndBiases(self, grads):
        self.Weights1   -= self.learningRate * grads["dWeights1"]
        self.bias1      -= self.learningRate * grads["dbias1"]
        self.Weights2   -= self.learningRate * grads["dWeights2"]
        self.bias2      -= self.learningRate * grads["dbias2"]

    def evaluate(self):
        # Perform forward propagation to get predictions
        forwardPropagationResult = self.forwardPropagation(self.X_forTesting)
        predictions = np.argmax(
            forwardPropagationResult["ActivationFunction_A2"], axis=1
        ) 

        # Calculate accuracy
        accuracy = (
            np.mean(predictions == self.Y_forTesting) * 100
        )  # Compare predictions to true labels and compute percentage

        return accuracy
