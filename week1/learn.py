
import pandas as pd
import numpy as np
from Model import Model 
from input import irisDataset


irisData = irisDataset()
Iris = Model()

Iris.prepareInputData(
    inputDataFrame = irisData, 
    numberOfClasses = 3, 
    numberOfFeatures = 4, 
    numberOfHiddenLayerUnits = 5, 
    splitRange = 120,
    learningRate = 0.073,
    epochs = 300
)

Iris.initializeStartRandomParameters()

for epoch in range(Iris.epochs):
    # Forward pass on training data
    forwardPropagationResults = Iris.forwardPropagation(Iris.X_forTraining)

    loss = Iris.compute_loss(Iris.Y_train_encoded, forwardPropagationResults["ActivationFunction_A2"])

    # Backward pass (compute gradients)
    grads = Iris.backwardPropagation(
        Iris.X_forTraining, Iris.Y_train_encoded, forwardPropagationResults
    )

    # Update parameters
    params = Iris.updateWeightsAndBiases(grads)

    # Print the loss every 100 epochs
    if epoch % 100 == 0:
        test_cache = Iris.forwardPropagation(Iris.X_forTesting)
        test_loss = Iris.compute_loss(Iris.Y_test_encoded, test_cache["ActivationFunction_A2"])

        print(f"Epoch {epoch}, Training loss: {loss:.4f}, Test loss: {test_loss:.4f}")
accuracy = Iris.evaluate()

print(f"Accuracy: {accuracy :.2f}%")

Iris.initializeStartRandomParameters()
