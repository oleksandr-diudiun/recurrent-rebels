from sklearn import datasets
import pandas as pd
import numpy as np


def irisDataset():
    # load iris dataset
    iris = datasets.load_iris()

    # since this is a bunch, create a dataframe

    iris_df = pd.DataFrame(iris.data)
    iris_df["class"] = iris.target
    iris_df.columns = ["sepal_len", "sepal_wid", "petal_len", "petal_wid", "class"]
    iris_df.dropna(how="all", inplace=True)  # remove any empty lines

    irisShuffled = iris_df.sample(frac=1).reset_index(drop=True)
    return irisShuffled
