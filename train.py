import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    csv = (
        "https://raw.githubusercontent.com/pontusnorman123/devops_tutorial/main/tutorialQuality.csv"
    )
    data = pd.read_csv(csv, sep=";")

    # Split the data into a training set and test set
    train, test = train_test_split(data)

    #Get prediction column
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    #Read model training parameters
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3

    mlflow.sklearn.autolog()
    
    with mlflow.start_run():
        
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)



